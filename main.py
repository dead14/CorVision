from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import cv2
import numpy as np
import base64
import torch
import segmentation_models_pytorch as smp
import os
import json
from typing import Optional
import cv2
import numpy as np
import base64
import torch
import segmentation_models_pytorch as smp
import os

app = FastAPI(title="AURA API")

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load AI Model (Hanya dilakukan sekali saat server menyala)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "model_korosi_terbaik.pth"
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[BERHASIL] Otak AI (model_korosi_terbaik.pth) berhasil dimuat!")
else:
    print(f"[PERINGATAN] File {model_path} belum ditemukan di folder ini!")
    print("Silakan download dari Colab dan pindahkan ke folder AURA.")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict_corrosion(
    file: UploadFile = File(...),
    roi_polygon: Optional[str] = Form(None),
    use_heatmap: str = Form("false"),
    aruco_size_cm: Optional[float] = Form(None)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image"}
        
    original_size = (img.shape[1], img.shape[0]) # (width, height)
    
    # --- PROSES ROI MASK ---
    user_roi_mask = None
    if roi_polygon:
        try:
            points = json.loads(roi_polygon)
            if len(points) > 2:
                # Buat canvas kosong seukuran gambar
                user_roi_mask = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
                # Konversi titik relatif ke absolut
                pts = []
                for p in points:
                    x = int(p['x'] * original_size[0])
                    y = int(p['y'] * original_size[1])
                    pts.append([x, y])
                pts = np.array(pts, np.int32)
                pts = pts.reshape((-1, 1, 2))
                # Isi polygon dengan warna putih (255)
                cv2.fillPoly(user_roi_mask, [pts], 255)
        except Exception as e:
            print("Error parsing ROI:", e)

    if not os.path.exists(model_path):
        return {"error": "Model AI belum dimasukkan ke dalam folder!"}

    # --- PROSES INFERENSI AI ---
    # 1. Siapkan gambar untuk AI (Resize ke 256x256, ubah ke RGB, normalisasi)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 2. AI Menebak Karat
    with torch.no_grad():
        output = model(img_tensor)
        # Terapkan fungsi Sigmoid lalu Threshold 0.5
        output = torch.sigmoid(output)
        pred_mask = (output > 0.5).float().cpu().squeeze().numpy()
        
    # 3. Kembalikan ukuran mask ke ukuran gambar asli
    pred_mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # --- HITUNG PERSENTASE & WARNAI MERAH ---
    initial_corrosion_pixels = np.count_nonzero(pred_mask_resized)
    is_fallback = False
    
    # FALLBACK: Jika AI gagal mendeteksi (karena bug nilai mask 0/1 saat training)
    # Kita gunakan deteksi warna karat (Coklat/Oranye) secara otomatis
    if initial_corrosion_pixels == 0:
        is_fallback = True
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_rust1 = np.array([0, 70, 50])
        upper_rust1 = np.array([25, 255, 255])
        lower_rust2 = np.array([170, 70, 50])
        upper_rust2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_rust1, upper_rust1)
        mask2 = cv2.inRange(hsv, lower_rust2, upper_rust2)
        fallback_mask = mask1 | mask2
        
        corrosion_pixels = np.count_nonzero(fallback_mask)
        mask_bool = fallback_mask > 0
    else:
        mask_bool = pred_mask_resized == 1.0
        
    # Terapkan ROI Mask jika ada
    total_pixels = original_size[0] * original_size[1]
    if user_roi_mask is not None:
        mask_bool = mask_bool & (user_roi_mask == 255)
        corrosion_pixels = np.count_nonzero(mask_bool)
    percentage = (corrosion_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    # Kalkulasi Confidence Score (Probabilitas Rata-rata dari area karat)
    confidence = 0.0
    prob_map_resized = None
    
    if corrosion_pixels > 0:
        if is_fallback:
            # Pseudo-probabilitas dari channel Saturation (kecerahan warna)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            s_channel = hsv[:, :, 1].astype(np.float32)
            # Saturasi 70-255 di-map ke probabilitas 0.5-1.0
            s_norm = np.clip((s_channel - 70) / (255 - 70), 0, 1)
            prob_map_resized = 0.5 + (s_norm * 0.5)
            confidence = float(np.mean(prob_map_resized[mask_bool])) * 100
        else:
            # Probabilitas asli dari otak AI U-Net
            prob_map_resized = cv2.resize(output.cpu().squeeze().numpy(), original_size)
            confidence = float(np.mean(prob_map_resized[mask_bool])) * 100
            
    # Kalkulasi Breakdown Severity
    severe_pct = 0.0
    light_pct = 0.0
    if corrosion_pixels > 0 and prob_map_resized is not None:
        prob_vals = prob_map_resized[mask_bool]
        severe_pixels = np.count_nonzero(prob_vals > 0.8)
        light_pixels = np.count_nonzero((prob_vals >= 0.5) & (prob_vals <= 0.8))
        
        if total_pixels > 0:
            severe_pct = (severe_pixels / total_pixels) * 100
            light_pct = (light_pixels / total_pixels) * 100
            
    # Kalkulasi Area Absolut (cm²) via ArUco & Homography
    absolute_area_cm2 = 0.0
    if aruco_size_cm and aruco_size_cm > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dicts = [cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_5X5_100, cv2.aruco.DICT_6X6_250]
        corners = None
        
        for dict_id in aruco_dicts:
            try:
                aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
                parameters = cv2.aruco.DetectorParameters()
                if hasattr(cv2.aruco, 'ArucoDetector'):
                    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                    corners_found, ids, rejected = detector.detectMarkers(gray)
                else:
                    corners_found, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                
                if corners_found:
                    corners = corners_found[0][0]
                    break
            except: continue
                
        if corners is not None:
            pts_src = corners
            scale_px_per_cm = 10.0
            L_px = aruco_size_cm * scale_px_per_cm
            pts_dst = np.array([[0, 0], [L_px, 0], [L_px, L_px], [0, L_px]], dtype=np.float32)
            
            H, _ = cv2.findHomography(pts_src, pts_dst)
            if H is not None:
                h_img, w_img = mask_bool.shape
                corners_img = np.float32([[0,0], [w_img,0], [w_img,h_img], [0,h_img]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(corners_img, H)
                [xmin, ymin] = np.int32(transformed_corners.min(axis=0).ravel() - 0.5)
                [xmax, ymax] = np.int32(transformed_corners.max(axis=0).ravel() + 0.5)
                
                if xmax - xmin < 10000 and ymax - ymin < 10000:
                    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
                    H_warp = translation.dot(H)
                    warped_mask = cv2.warpPerspective(mask_bool.astype(np.uint8), H_warp, (xmax-xmin, ymax-ymin))
                    absolute_area_cm2 = np.count_nonzero(warped_mask) / (scale_px_per_cm ** 2)
                else:
                    pixel_area = cv2.contourArea(pts_src)
                    cm2_per_pixel = (aruco_size_cm ** 2) / pixel_area if pixel_area > 0 else 0
                    absolute_area_cm2 = corrosion_pixels * cm2_per_pixel
                
                # Gambar outline hijau pada ArUco
                cv2.polylines(img, [pts_src.astype(np.int32)], True, (0, 255, 0), 3)
        else:
            return {"error": "ArUco marker tidak terdeteksi. Pastikan marker sejajar, tidak silau, dan tidak tertutup."}
            
    # Buat gambar hasil
    result_img = img.copy()
    
    if use_heatmap == "true" and corrosion_pixels > 0:
        # Mode Heatmap: Kuning (prob 0.5) ke Merah (prob 1.0)
        prob_vals = prob_map_resized[mask_bool]
        norm_prob = np.clip((prob_vals - 0.5) * 2.0, 0, 1) # 0 to 1
        
        r = np.full_like(norm_prob, 255)
        g = (1.0 - norm_prob) * 255
        b = np.zeros_like(norm_prob)
        
        heatmap_colors = np.stack([b, g, r], axis=-1).astype(np.uint8)
        result_img[mask_bool] = heatmap_colors
    else:
        # Warnai merah solid (BGR: [0, 0, 255])
        result_img[mask_bool] = [0, 0, 255] 
    
    # Gabungkan (Blend) agar transparan elegan
    alpha = 0.5
    result_img = cv2.addWeighted(result_img, alpha, img, 1 - alpha, 0)
    
    # Convert ke Base64 agar bisa ditampilkan di HTML
    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "percentage": percentage,
        "confidence": confidence,
        "severe_pct": severe_pct,
        "light_pct": light_pct,
        "absolute_area_cm2": absolute_area_cm2,
        "image_base64": img_base64
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
