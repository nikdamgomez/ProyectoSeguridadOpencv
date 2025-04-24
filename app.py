from flask import Flask, render_template , Response
import cv2
from roboflow import Roboflow
import os
from ultralytics import YOLO
from pathlib import Path
import threading





app = Flask(__name__)

def gen_frames():

    # Load your trained model
    model = YOLO("runs/detect/train5/weights/best.pt")  # Update path if needed


    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Run detection
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cls = int(box.cls[0])
                label = model.names[cls]

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode and yield the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

def train():
    # ¡Reemplaza esta clave con tu propia API key de Roboflow!
    # Puedes encontrar tu API key en https://app.roboflow.com/settings
    with open("validate.txt", "r") as file:
        content = file.readline().strip()
        if content == "True":
            print("Validation is True. Exiting function early.")
            return  # Stops the function here

    rf = Roboflow(api_key="HGpn9sxgRvVoq6BS6pDB")

    # ¡Reemplaza con el nombre de tu espacio de trabajo y el nombre de tu proyecto!
    workspace = rf.workspace("prototipo-rro16")
    project = workspace.project("safety-vest---v4-he0au")

    # ¡Reemplaza con el número de la versión que quieres descargar y el formato ("yolov8")!
    # Puedes ver las versiones de tu dataset en la página de tu proyecto en Roboflow
    version = project.version(1)
    dataset = version.download("yolov8")

    print(f"El dataset y el modelo se han descargado en: {dataset.location}")


    project = rf.workspace("prototipo-rro16").project("safety-vest---v4-he0au")
    dataset = project.version(1).download("yolov8")

    ruta_dataset = 'Safety-vest---v4-1'

    # Update the path to your downloaded dataset folder:
    dataset_folder = 'Safety-vest---v4-1'

    # Construct the correct paths for train and valid folders:
    train_folder = dataset_folder+ '/train'
    valid_folder = dataset_folder+ '/valid'

    # Now, list the files in the correct directories:
    print("Archivos en train:", os.listdir(train_folder))
    print("Archivos en valid:", os.listdir(valid_folder))

    # Assuming your dataset is downloaded to 'Safety-vest---v4-1'
    dataset_folder = 'Safety-vest---v4-1'
    # Update this path if it is different.

    # Construct the correct path to your data.yaml file:
    data_yaml_path = dataset_folder+ '/data.yaml'

    # Now, open the file:
    with open(data_yaml_path, "r") as file:
        print(file.read())

    print("Contenido de la carpeta actual:", os.listdir("."))
    if os.path.exists("seguridad-1"):
        print("Contenido de seguridad-1:", os.listdir("seguridad-1"))
        if os.path.exists("seguridad-1/valid"):
            print("Contenido de seguridad-1/valid:", os.listdir("seguridad-1/valid"))
        if os.path.exists("seguridad-1/train"):
            print("Contenido de seguridad-1/train:", os.listdir("seguridad-1/train"))
        
    model = YOLO("yolov8n.pt")

    # Update the data path to the correct location of your data.yaml file
    # Replace with the actual path to your data.yaml file:
    data_yaml_path = 'Safety-vest---v4-1/data.yaml'
    results = model.train(data=data_yaml_path, epochs=30, imgsz=640)
    #results = model.train(data=data_yaml_path, epochs=30, imgsz=416)
    with open("validate.txt", "w") as file:
        file.write("True")


            


@app.route('/')
def index ():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__": 
 threading.Thread(target=train).start()
 app.run(debug=True)
