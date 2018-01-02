# chess_detection

## About
For chessboard grid detection:
- X-corner detection using [Automatic Chessboard Corner Detection Method](https://www.researchgate.net/publication/282446068_Automatic_chessboard_corner_detection_method)
- Iterative grid fitting from [this](https://github.com/Elucidation/ChessboardDetect/blob/master/FindChessboards.py)

For piece detection:
- Data generation using LabalImg (a lot of time spent)
- Fine tune pretrained [faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2017_11_08.tar.gz) from object detection api

## Dependencies (not too sure)

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

* Download my [fine tuned model]() (waiting upload...) and optionally [test images]() (waiting upload...). Extract them in the main repo folder.


For simgle images:
```shell
python gui.py
```
For bulk testing of test images:
```shell
python test.py
```

## Example result
<div align="center">
  <img src="https://github.com/jonkoi/chess_detection/blob/master/sample_result.png"><br><br>
</div>

[Here is video of the test.]()

## Disclaimer
The piece detection module uses deep learning (not mobilenet) which can make inference very slow. The process is sluggish even with a GTX860M.
