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

```
