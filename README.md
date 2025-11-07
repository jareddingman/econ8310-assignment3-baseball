# Assignment 3 (Jared Dingman and Zach Edwards)
## Econ 8310 - Business Forecasting

This is a repo for taking exported CVAT xml files in conjunction with raw baseball videos to make "Baseball" predictions. Dusty, please see the file marked fullAssingment.py for our graded pytorch dataloder and model.

To run the model:
- specify your raw video directory
- specify your annotations file directory

Note that the neural net can be ran with the other file marked LoadModel.py if desired.

Some checks you can make on your own:
- Can your custom loader import a new video or set of videos?
- Does your script train a neural network on the assigned data?
- Did your script save your model?
- Do you have separate code to import your model for use after training?
