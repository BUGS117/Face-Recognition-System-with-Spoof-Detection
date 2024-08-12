import face_recognition
import asyncio
import websockets
import json
import io
import numpy as np
import os

faces = []
files = os.listdir('faces')

for file in files:
    faces.append(np.load("faces/" + file))


async def websocket_handler(websocket):
    try:
        async for message in websocket:
            response = data_extract(message)
            await websocket.send(json.dumps(response))
    except Exception as e:
        print(f"WebSocket Error: {str(e)}")


def data_extract(message):
    try:
        json_data = json.loads(message)
        img = bytes(json_data['image'])
        unknown_picture = face_recognition.load_image_file(io.BytesIO(img))

        unknown_face_encodings = face_recognition.face_encodings(unknown_picture)
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
        results = face_recognition.compare_faces(faces, unknown_face_encoding, 0.4)

        success = False
        for result in results:
            if result:
                success = True
                break

        if success:
            index = results.index(True)
            payCode = files[index].split(".npy")[0]
            return {"status": True, "message": "Recognition successful", "payCode": payCode, "data": 1}
        else:
            return {"status": True, "message": "Recognition unsuccessful", "data": 0}

    except Exception as e:
        return {"status": False, "message": str(e), "data": 0}


def face_reg(unknown_face_encoding, json_data):
    try:
        payCode = json_data["payCode"]
        file_path = "faces/" + payCode + ".npy"
        np.save(file_path, unknown_face_encoding)
        return {"status": True, "message": file_path, "data": 1}
    except Exception as e:
        return {"status": False, "message": str(e), "data": 0}


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        websockets.serve(websocket_handler, "192.168.0.194", 8765)
    )
    loop.run_forever()
