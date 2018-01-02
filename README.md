# chess_detection

### Dependencies (not too sure)

* Python 3.6
* OpenCV3
* tensorflow including tensorflow's models repo
* protoc
* PIL
* numpy
* scipy
* svglib
* reportlab
* pandas
* ...

## How to use

```shell
git clone git@github.com:jonkoi/chess_detection.git
cd chess_detection
git clone git@github.com:tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.
set PYTHONPATH=PATH\TO\models\research;PATH\TO\models\research\slim (for windows)
cd ../..
```

* Download my [fine tuned model] of [faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2017_11_08.tar.gz) and optionally [test images]. Extract them in the main repo folder.

```shell
python gui.py
```

## Example result
