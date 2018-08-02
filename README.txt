Prototype 1
load models, read labels
ask user to upload an image
store the image locally
process and make predictions
plot the output
delete the input image
display the result of prediction

to run:
start up anaconda, click environments, then select your flask environment, click on the green triangle at the side and click open Terminal
this should open up cmd, then enter cd to your work directory which contain package folder and setup.py
then enter
set FLASK_APP=package
flask run

then once it finishes loading the models etc visit http://127.0.0.1:5000/ on your browser to see the product