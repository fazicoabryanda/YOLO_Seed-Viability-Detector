from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from PIL import Image
import cv2

app = Flask(__name__)

# Path ke model YOLOv8
model = YOLO('best.pt')

# Path untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Membuat folder upload jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Halaman Utama
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk mengunggah gambar
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Melakukan prediksi dengan YOLO, sembunyikan label dan confidence score
        results = model(filepath, hide_labels=True, hide_conf=True)

        # Load gambar asli
        img = cv2.imread(filepath)

        # Warna untuk bounding box
        color_viable = (0, 255, 0)  # Hijau untuk "viable"
        color_non_viable = (0, 0, 255)  # Merah untuk "non-viable"

        # Hitung jumlah viable dan non-viable
        viable_count = 0
        non_viable_count = 0

        # Gambar bounding box dan hitung jumlah
        for result in results[0].boxes:
            label = int(result.cls)
            box = result.xyxy[0].cpu().numpy().astype(int)  # Koordinat bounding box
            
            if label == 0:  # Viable
                viable_count += 1
                color = color_viable
            elif label == 1:  # Non-viable
                non_viable_count += 1
                color = color_non_viable
            
            # Gambarkan kotak pembatas
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Tambahkan keterangan warna di gambar
        cv2.putText(img, 'Viable (Hijau)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_viable, 2, cv2.LINE_AA)
        cv2.putText(img, 'Non-Viable (Merah)', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color_non_viable, 2, cv2.LINE_AA)

        # Simpan gambar hasil prediksi
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
        cv2.imwrite(output_path, img)

        # Kembalikan hasil sebagai JSON
        return jsonify({
            'uploaded_image': filename,
            'result_image': 'result_' + filename,
            'viable_count': viable_count,
            'non_viable_count': non_viable_count
        })

if __name__ == '__main__':
    app.run(debug=True)
