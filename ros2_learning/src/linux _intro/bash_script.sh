#!/bin/bash

ARG1=$1

if [ "$ARG1" == "circle" ]; then
   echo "circling";
   rosrun move_bb8_pkg move_bb8_circle.py 

elif [ "$ARG1" == 'forward_backward' ]; then
     echo "back and forth";
     rosrun move_bb8_pkg move_bb8_forward_backward.py

elif [ "$ARG1" == "square" ]; then
     echo "square dancing";
     rosrun move_bb8_pkg move_bb8_square.py

else 
echo "Please enter one of the following;
circle
forward_backward
square"

fi
