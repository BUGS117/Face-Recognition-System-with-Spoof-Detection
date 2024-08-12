import asyncio
import websockets
import json
import io
import os
import pickle
import face_recognition
from PIL import Image
import numpy as np

db_dir = './faces'

if not os.path.exists(db_dir):
    os.mkdir(db_dir)

async def websocket_handler(websocket):
    try:
        async for message in websocket:
            response = await data_extract(message)
            await websocket.send(json.dumps(response))
    except Exception as e:
        print(f"WebSocket Error: {str(e)}")

async def data_extract(message):
    try:
        json_data = json.loads(message)
        img = bytes(json_data['image'])
        unknown_picture = face_recognition.load_image_file(io.BytesIO(img))

        # Detect face locations
        face_locations = face_recognition.face_locations(unknown_picture)
        if len(face_locations) == 0:
            return {"status": True, "message": "No Face Detected", "data": 0}
        
        # Use the first face detected
        top, right, bottom, left = face_locations[0]
        face_image = unknown_picture[top:bottom, left:right]

        # Resize the cropped face image to a fixed width and height (e.g., 800x800)
        pil_image = Image.fromarray(face_image)
        pil_image = pil_image.resize((800, 800))
        face_image = np.array(pil_image)

        unknown_face_encodings = face_recognition.face_encodings(face_image)
        if len(unknown_face_encodings) > 0:
            unknown_face_encoding = unknown_face_encodings[0]
            if json_data['type'] == 'reg':
                return face_reg(unknown_face_encoding, json_data)
            else:
                return recognize_face(unknown_face_encoding)
        else:
            return {"status": True, "message": "No Face Detected", "data": 0}
    except Exception as e:
        return {"status": False, "message": str(e), "data": 0}

def recognize_face(unknown_face_encoding):
    try:
        files = sorted(os.listdir(db_dir))
        match = False
        for file in files:
            path_ = os.path.join(db_dir, file)
            with open(path_, 'rb') as f:
                known_face_encoding = pickle.load(f)
                match = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding, 0.4)[0]
                if match:
                    name = file.split('.pickle')[0]
                    return {"status": True, "message": "Recognition successful", "payCode": name, "data": 1}
        
        return {"status": True, "message": "Recognition unsuccessful", "data": 0}

    except Exception as e:
        return {"status": False, "message": str(e), "data": 0}

def face_reg(unknown_face_encoding, json_data):
    try:
        payCode = json_data["payCode"]
        file_path = os.path.join(db_dir, f"{payCode}.pickle")
        with open(file_path, 'wb') as f:
            pickle.dump(unknown_face_encoding, f)
        return {"status": True, "message": file_path, "data": 1}
    except Exception as e:
        return {"status": False, "message": str(e), "data": 0}

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        websockets.serve(websocket_handler, "192.168.0.194", 8765)
    )
    loop.run_forever()