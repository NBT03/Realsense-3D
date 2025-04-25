
from roboflow import Roboflow
rf = Roboflow(api_key="r12RUnLPDBmqrHbJSnn9")
project = rf.workspace("yolo-macse").project("shapes-aprto-a5uaq")
version = project.version(2)
dataset = version.download("yolov11")
                
