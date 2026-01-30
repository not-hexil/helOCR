import http.server
import json
import os
from ocr import OCRNeuralNetwork

nn = OCRNeuralNetwork(15, [], [], [], use_file=True)

class OCRRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/ocr.html':
            self.path = '/ocr.html'
            content_type = 'text/html'
        elif self.path.endswith('.js'):
            content_type = 'application/javascript'
        elif self.path.endswith('.css'):
            content_type = 'text/css'
        else:
            self.send_error(404, "File not found")
            return

        try:
            file_path = self.path[1:]
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "File not found")
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")

    def do_POST(self):
        response_code = 200
        response = ""
        
        try:
            content_len = int(self.headers.get('Content-Length', 0))
            if content_len == 0:
                self.send_error(400, "Empty request body")
                return
                
            content = self.rfile.read(content_len)
            payload = json.loads(content)

            if payload.get('train'):
                train_data = {
                    'y0': payload.get('trainArray')[0]['y0'],
                    'label': payload.get('trainArray')[0]['label']
                }
                nn.train(train_data)
                nn.save()
                response = {"type": "train", "status": "success"}
                
            elif payload.get('predict'):
                image_data = payload['image']
                print(f"DEBUG: Image data type: {type(image_data)}")
                if isinstance(image_data, list):
                    print(f"DEBUG: Image data length: {len(image_data)}")
                    if len(image_data) > 0:
                        print(f"DEBUG: First element: {image_data[0]} ({type(image_data[0])})")
                
                prediction = nn.predict(image_data)
                response = {
                    "type": "test", 
                    "result": str(prediction)
                }
            else:
                response_code = 400
                response = {"error": "Invalid request type"}
                
        except json.JSONDecodeError:
            print("JSON Decode Error")
            response_code = 400
            response = {"error": "Invalid JSON"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            response_code = 500
            response = {"error": f"Server error: {str(e)}"}

        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        if response:
            self.wfile.write(json.dumps(response).encode('utf-8'))

if __name__ == '__main__':
    PORT = 8000
    server_address = ('', PORT)
    httpd = http.server.HTTPServer(server_address, OCRRequestHandler)
    print(f"Starting OCR Server on http://localhost:{PORT}")
    print(f"Open http://localhost:{PORT}/ocr.html in your browser")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()