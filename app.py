import os
import io
import time
import json
import base64
import sqlite3
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functools import wraps
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask import (
    Flask, request, jsonify, render_template,
    session, redirect, url_for, flash
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "contrail_model.pth")
HIST_PATH  = os.path.join(BASE_DIR, "history.json")
DB_PATH    = os.path.join(BASE_DIR, "users.db")
IMAGE_SIZE = 256
THRESHOLD  = 0.35
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── LOAD MODEL (9-channel + scSE attention — matches model_1.py) ─────────────
print(f"[SkyTrace] Loading model on {DEVICE}…")
_model = smp.Unet(
    encoder_name   = "efficientnet-b3",
    encoder_weights= None,
    in_channels    = 9,
    classes        = 1,
    activation     = None,
    decoder_attention_type = "scse",
)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
_model.load_state_dict(checkpoint["model_state"])
_model.to(DEVICE)
_model.eval()
print(f"[SkyTrace] Model ready — Dice={checkpoint['best_dice']:.4f}, "
      f"Epoch={checkpoint['epoch']}")

# ── USER AUTHENTICATION ────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE & AUTH
# ══════════════════════════════════════════════════════════════════════════════
from database import init_db, get_db, login_required


# ── MODEL HELPERS ─────────────────────────────────────────────────────────────
def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image → (1, 9, H, W) float32 tensor.
    Duplicates RGB 3× to simulate temporal frames for 9-channel input.
    """
    img = pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr9 = np.concatenate([arr, arr, arr], axis=-1)
    tensor = torch.from_numpy(arr9.transpose(2, 0, 1))
    return tensor.unsqueeze(0)


def run_inference(pil_img: Image.Image):
    """Run full inference pipeline. Returns probs, mask, orig_arr, metrics."""
    t0 = time.time()
    tensor = preprocess(pil_img).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.sigmoid(logits.float()).squeeze().cpu().numpy()

    mask = (probs > THRESHOLD).astype(np.float32)
    elapsed_ms = (time.time() - t0) * 1000

    total_px    = mask.size
    contrail_px = int(mask.sum())
    coverage    = round(contrail_px / total_px * 100, 2)
    conf        = round(float(probs[mask == 1].mean()) if contrail_px > 0 else 0.0, 4)

    orig_arr = np.array(
        pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR),
        dtype=np.float32
    ) / 255.0

    metrics = {
        "coverage_pct":    coverage,
        "mean_confidence": conf,
        "processing_ms":   round(elapsed_ms, 1),
        "threshold":       THRESHOLD,
        "contrail_pixels": contrail_px,
        "total_pixels":    total_px,
    }
    return probs, mask, orig_arr, metrics


def pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def heatmap_b64(probs: np.ndarray, cmap_name: str = "Blues") -> str:
    cmap = cm.get_cmap(cmap_name)
    rgba = (cmap(probs) * 255).astype(np.uint8)
    img  = Image.fromarray(rgba, "RGBA").convert("RGB")
    return pil_to_b64(img)


def probmap_b64(probs: np.ndarray) -> str:
    gray = (probs * 255).astype(np.uint8)
    return pil_to_b64(Image.fromarray(gray, "L"))


def overlay_b64(orig: np.ndarray, mask: np.ndarray, probs: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(orig)
    ax.imshow(probs, cmap="Blues", alpha=0.55, vmin=0, vmax=1)
    if mask.sum() > 0:
        ax.contour(mask, levels=[0.5], colors=["cyan"], linewidths=[1.2])
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_response(pil_img: Image.Image, airspace_info=None):
    probs, mask, orig_arr, metrics = run_inference(pil_img)
    resp = {
        "original": pil_to_b64(Image.fromarray((orig_arr * 255).astype(np.uint8))),
        "heatmap":  heatmap_b64(probs),
        "overlay":  overlay_b64(orig_arr, mask, probs),
        "probmap":  probmap_b64(probs),
        "metrics":  metrics,
    }
    if airspace_info:
        resp["airspace"] = airspace_info
    return resp


# ── FLASK APP ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "skytrace-secret-key-2026-contrail-optimizer"
app.config["PERMANENT_SESSION_LIFETIME"] = 86400  # 24 hours

# Configure secure session cookies to work inside Hugging Face iframes
app.config.update(
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=True
)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page and handler."""
    # If already logged in, go to dashboard
    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please fill in all fields.", "error")
            return render_template("login.html")

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        db.close()

        if user and check_password_hash(user["password_hash"], password):
            session.permanent = True
            session["user_id"]   = user["id"]
            session["user_name"] = user["full_name"]
            session["user_email"] = user["email"]
            session["user_role"]  = user["role"]
            session["user_org"]   = user["organization"]
            return redirect(url_for("index"))
        else:
            flash("Invalid email or password.", "error")
            return render_template("login.html")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Signup page and handler."""
    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        full_name    = request.form.get("full_name", "").strip()
        email        = request.form.get("email", "").strip().lower()
        password     = request.form.get("password", "")
        confirm      = request.form.get("confirm_password", "")
        role         = request.form.get("role", "Flight Dispatcher")
        organization = request.form.get("organization", "").strip()

        # Validation
        if not full_name or not email or not password:
            flash("Please fill in all required fields.", "error")
            return render_template("signup.html")

        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("signup.html")

        # Create user
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (full_name, email, password_hash, role, organization) VALUES (?, ?, ?, ?, ?)",
                (full_name, email, generate_password_hash(password), role, organization)
            )
            db.commit()
            db.close()

            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            db.close()
            flash("An account with this email already exists.", "error")
            return render_template("signup.html")

    return render_template("signup.html")


@app.route("/logout")
def logout():
    """Clear session and redirect to login."""
    session.clear()
    return redirect(url_for("login"))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTES (Protected)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def index():
    return render_template("index.html",
        user_name  = session.get("user_name", "User"),
        user_email = session.get("user_email", ""),
        user_role  = session.get("user_role", "Dispatcher"),
        user_org   = session.get("user_org", ""),
    )


@app.route("/stats")
def stats():
    return jsonify({
        "encoder":      "EfficientNet-B3",
        "architecture": "U-Net + scSE Attention",
        "best_dice":    checkpoint["best_dice"],
        "best_epoch":   checkpoint["epoch"],
        "threshold":    THRESHOLD,
        "device":       DEVICE,
        "image_size":   IMAGE_SIZE,
        "in_channels":  9,
        "loss":         "0.5×Dice + 0.5×Focal",
        "dataset":      "Google Research – Identify Contrails (Kaggle)",
    })


@app.route("/history")
def history():
    try:
        with open(HIST_PATH) as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "history.json not found"}), 404





@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        pil_img = Image.open(f.stream)
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    airspace_meta = {"name": f.filename, "code": "USR", "flights": "—"}
    return jsonify(build_response(pil_img, airspace_meta))


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 7860))
    print("\n" + "=" * 55)
    print("  SkyTrace — Flight Route Optimizer")
    print(f"  http://localhost:{port}")
    print("=" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=port)
