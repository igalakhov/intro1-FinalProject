extensions [matrix]
globals [
 inputs ;;pcolor inputs
 outputs ;;correct labels for inputs
 theta1 ;;first theta
 theta2 ;;second theta
 batch-correct ;;current error on neural net
]
turtles-own [
 centroidId
]
patches-own [
 which-centroid
 org-color
 c0-col
 c1-col
]
to setup
  reset-ticks
  set theta1 (create-theta hidden-layer-size 401)
  set theta2 (create-theta 10 (hidden-layer-size + 1))
end
to-report test-cur
  let cur (matrix:make-constant 1 400 0)
  matrix:set-row cur 0 get-patch-colors

  let a1 (matrix:copy (add-ones (matrix:copy cur)))
  let z2 (a1 matrix:* (matrix:transpose theta1))
  let a2 (add-ones (sigmoid (matrix:copy z2)))
  let z3 (a2 matrix:* (matrix:transpose theta2))
  let a3 (sigmoid z3)

  let curMax (max item 0 matrix:to-row-list a3)
  let prediction position curMax item 0 matrix:to-row-list a3
  report prediction
end
to setup-two
  cp
  resize-world 0 25 -19 0
end
to setup-one
  cp
  resize-world 0 19 -19 0
end
;;attempts to use K means clustering
to-report classify-two
  let k-threshold 10
  cro 1 [setxy 0 (-19 / 2) set centroidId 0 set color red]
  cro 1 [setxy 19 (-19 / 2) set centroidId 1 set color blue]
  ask patches [set which-centroid -1 set org-color pcolor]
  repeat k-threshold [
    ;;classify
    ask patches with [pcolor != black] [
      set which-centroid (item 0 ([centroidId] of turtles with-min [distance myself]))
      ;;show which-centroid
      ifelse (which-centroid = 0)[set pcolor red][set pcolor blue]
    ]
    ;;move turtles
    ask turtles with [centroidId = 0][
      carefully [
      set xcor mean [pxcor] of patches with [pcolor = red]
      set ycor mean [pycor] of patches with [pcolor = red]
      ][]
    ]
    ask turtles with [centroidId = 1][
      carefully [
      set xcor mean [pxcor] of patches with [pcolor = blue]
      set ycor mean [pycor] of patches with [pcolor = blue]
      ][]
    ]
    ;reset
    ask patches with [pcolor = blue or pcolor = red] [set pcolor white]
  ]
  ask turtles [die]
  ask patches [if which-centroid = 0 [set pcolor blue] if which-centroid = 1 [set pcolor red]]
  ;;isolate the digits and classify them separately
  let avgX0 0
  let avgY0 0
  let avgX1 0
  let avgY1 0
  carefully [
  set avgX0 (19 / 2) - ((min [pxcor] of patches with [which-centroid = 0]) + (max [pxcor] of patches with [which-centroid = 0])) / 2
  set avgY0 (-19 / 2) - ((min [pycor] of patches with [which-centroid = 0]) + (max [pycor] of patches with [which-centroid = 0])) / 2
  set avgX1 (19 / 2) - ((min [pxcor] of patches with [which-centroid = 1]) + (max [pxcor] of patches with [which-centroid = 1])) / 2
  set avgY1 (-19 / 2) - ((min [pycor] of patches with [which-centroid = 1]) + (max [pycor] of patches with [which-centroid = 1])) / 2
  ][]
  ask patches with [which-centroid = 0] [
    ask patch-at avgX0 avgY0 [set c0-col white]
  ]
  ask patches with [which-centroid = 1] [
    ask patch-at avgX1 avgY1 [set c1-col white]
  ]
  ask patches [set pcolor c0-col]
  wait 1
  let tensPlace test-cur
  ask patches [set pcolor c1-col]
  wait 1
  let onesPlace test-cur
  ask patches [set pcolor org-color]
  wait 1
  report (10 * tensPlace) + onesPlace
end
to load-preset-weights
  ;;loads preset weights from text file
  file-close-all
  file-open "trainedWeights.txt"
  set theta1 (matrix:from-row-list file-read)
  set theta2 (matrix:from-row-list file-read)
  file-close-all

end
to train-once
  ;;make training set
  create-training-set batch-size

  ;;forward propagate
  let a1 (matrix:copy (add-ones (matrix:copy inputs)))
  let z2 (a1 matrix:* (matrix:transpose theta1))
  let a2 (add-ones (sigmoid (matrix:copy z2)))
  let z3 (a2 matrix:* (matrix:transpose theta2))
  let a3 (sigmoid z3)

  ;;update error
  set batch-correct (correct-in-batch a3 outputs)

  ;;backpropagate
  let d3 (a3 matrix:- outputs)
  let d2 (matrix:times-element-wise (d3 matrix:* (delete-first-column theta2)) (sigmoid-gradient z2))

  ;;calculate slopes
  let delta1 (matrix:times-scalar ((matrix:transpose d2) matrix:* a1) (1 / batch-size))
  let delta2 (matrix:times-scalar ((matrix:transpose d3) matrix:* a2) (1 / batch-size))

  set delta1 (matrix:map apply-learning-rate delta1)
  set delta2 (matrix:map apply-learning-rate delta2)

  ;;apply slopes
  set theta1 (theta1 matrix:- delta1)
  set theta2 (theta2 matrix:- delta2)

end
to-report apply-learning-rate [n]
  report n * learning-rate
end
;;reports error between predicted and actual values of a neural network
to-report correct-in-batch [predicted actual]
  set predicted (matrix:to-row-list predicted)
  set actual (matrix:to-row-list actual)

  let numCorrect 0

  let i 0
  while [i < (length predicted)][
   let curPredicted (item i predicted)
   let curActual (item i actual)

   set curPredicted position (max curPredicted) curPredicted
   set curActual position 1 curActual

   if(curPredicted = curActual) [
     set numCorrect (numCorrect + 1)
   ]

   set i (i + 1)
  ]
  report (numCorrect / (length predicted))
end
;;function used in backpropagation
to-report delete-first-column [matrixIn]
  report (matrix:from-column-list (but-first (matrix:to-column-list (matrix:copy matrixIn))))
end
;;derivative of the sigmoid function
to-report sigmoid-gradient [matrixIn]
  report (matrix:copy (matrix:map sigmoid-gradient-one (matrix:copy matrixIn)))
end
to-report sigmoid-gradient-one [n]
  report ((sigmoid-one n) * (1 - (sigmoid-one n)))
end
to-report sigmoid [matrixIn]
  report (matrix:copy (matrix:map sigmoid-one (matrix:copy matrixIn)))
end
to-report sigmoid-one [n]
  report (1 / (1 + (exp (-1 * n))))
end
;;adds ones to the left side of a matrix
to-report add-ones [matrixIn]
  report (matrix:from-column-list (fput (ones (item 0 (matrix:dimensions matrixIn)))(matrix:to-column-list (matrix:copy matrixIn))))
end
to-report ones [len]
  let out []
  let i 0
  while [i < len][
   set out (fput 1 out)
   set i (i + 1)
  ]
  report out
end
;; creates a starting theta
;; we can't just let the theta be the same number because
;; that cause the network to become parallel
;; which is not good
to-report create-theta [height width]
  let out (matrix:map randValue (matrix:make-constant height width 0))
  report out
end
;; helper function for create-theta
to-report randValue [n]
  report (((random 11) / 10) - 0.5)
end
to create-training-set [n]
  set inputs (matrix:make-constant n 400 0)
  set outputs (matrix:make-constant n 10 0)

  let i 0
  ;;generate data
  while [i < n][
      let curNum (random 10)
      load-random curNum
      matrix:set-row inputs i get-patch-colors
      matrix:set-row outputs i (create-lable curNum)
      tick
      set i (i + 1)
  ]
end
;;reports a lable for a number
to-report create-lable [n]
  report replace-item n (zeros 10) 1
end
;;reports a list of zeros with size n
to-report zeros [len]
  let out []
  let i 0
  while [i < len][
   set out (fput 0 out)
   set i (i + 1)
  ]
  report out
end
;;collects patches pcolors and reports back a list of size 400
;;pcolors are scaled from 0 to 1, since this is grayscale
to-report get-patch-colors
  let out (matrix:make-constant 1 400 0)
  let y 0
  while [y >= -19][
   let x 0
   while [x <= 19][
      let curColor [pcolor] of patch x y
      ;;ask patch x y [set plabel (x + ((abs y) * 20))]
      matrix:set out 0 (x + ((abs y) * 20)) (curColor / 10)
   set x (x + 1)
   ]
   set y (y - 1)
  ]
  report item 0 (matrix:to-row-list (matrix:copy out))
end
to setup-draw
  cp
end
to draw
  if (mouse-down?) [
   ask patch mouse-xcor mouse-ycor[
     set pcolor white
     ask neighbors4 [
       if (pcolor != white) [
          set pcolor white
          set pcolor random 10
        ]
      ]
    ]
  ]
end
;;loads a random digit based on number in
to load-random [num]
  cp
  file-open (word (word "testing/" num) "/desc.txt")
  let num-examples read-from-string file-read-line
  file-close

  let rand ((random num-examples) + 1)
  ;;show word word word word "testing/" num "/(" rand ").png"
  import-pcolors word word word word "testing/" num "/(" rand ").png"
end
@#$#@#$#@
GRAPHICS-WINDOW
227
14
646
434
-1
-1
20.55
1
10
1
1
1
0
0
0
1
0
19
-19
0
0
0
1
ticks
30.0

BUTTON
782
462
994
495
Load Random Digit From Dataset
load-random (random 10)\n
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
784
62
879
95
NIL
setup-draw\n
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
784
97
847
130
NIL
draw
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

TEXTBOX
783
18
933
43
User buttons
20
0.0
1

BUTTON
73
57
137
90
Setup
ca\nsetup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
75
308
138
341
Train
train-once
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
782
421
997
454
Classify Current Drawing (1 digit)
user-message test-cur
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
28
117
200
150
hidden-layer-size
hidden-layer-size
1
100
100.0
1
1
NIL
HORIZONTAL

BUTTON
63
262
152
295
Train once
train-once
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
31
165
203
198
batch-size
batch-size
1
10000
51.0
50
1
NIL
HORIZONTAL

SLIDER
30
219
202
252
learning-rate
learning-rate
0
1
1.0
0.01
1
NIL
HORIZONTAL

BUTTON
779
211
1029
244
Load Preset Weights ( ~95% accuracy)
load-preset-weights\n
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
782
289
969
322
Setup two digit classification
setup-two
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
783
332
1002
365
Classiffy Current Drawing (2 digit)
user-message classify-two\n
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
782
377
969
410
Setup one digit classification
setup-one
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

PLOT
13
364
213
611
% correct in last batch
Time
% correct in last batch
0.0
10.0
0.0
100.0
true
false
"" ""
PENS
"" 1.0 0 -16777216 true "" "plot batch-correct * 100"

@#$#@#$#@
## WHAT IS IT?

This is a netlogo simulation that allows you to train a neural network that can reconize and classify hand written digits. 

Read below if you don't know what neural networks are. 

## WHAT ARE NEURAL NETWORKS?

A neural network is a mathematical model that behaves like the human brain. Using a large network of neurons trained to behave in a certain way under certain conditions, a neural network can perform very complicated tasks, in this case classifying handwritten digits. 

Lets first talk about the simplest component of a neural network - a neuron. 

![neuron](neuron.png)

A neuron takes in a set of inputs (usually scaled from 0 to 1) and, from that, generates an output. This output is usally created by multiplying each input by a certain weight, summing these products together, and feeding this sum through a special function that scales it between 0 and 1. Chaining many neurons together creates a neural network:

![neuron](sampleNetwork.png)

A neural network is organized into rows of neurons called layers. Each neuron in a layer uses the outputs of the previous layer as its inputs. The only exception to this is the first layer. The inputs to this layer are controlled by the user of the network. In the case of the network used in this simulation, the first layer has 400 neurons, each representing the scaled pcolor of the 20x20 drawing that we are trying to classify. 

The last layer of the network is called the output layer. The output of neurons in this layer tells us what the neural network "thinks" about out input. The network in this model has 10 neurons in this layer, each representing a digit from 0 to 9. Once an example has been fed through a network, all of these neurons have outputs scaled from 0 to 1. The higher this ouput is, the more sure the network is about the input representing the given digit. To classify an input, the neuron with the highest value is chosen. 

The middle layer of the network is called the hidden layer. This layer is what allows the network to perform very complicated tasks, as it creates an extra transfer of information between the input and output layers. The network in this simulation only has one, but networks used for other purposes can theoretically have as many as needed.

Neural networks are trained by adjusting the weights of its neurons. This is done using a special algorithm known as backpropagation. Given a set of inputs and their correct outputs, backpropagation allows us to calculate the weights' "errors", and adjust them to be slightly more correct. A neural network is usally trained in batches, where a large number of inputs is fed through the network before the network is moved in the right direction using the average error between all those inputs. 


## HOW TO TRAIN A NETWORK

1. Press the setup button. This will initialize all internal variables, and load the network with starting weights. 
2. Configure the network parameters. Make sure to press the setup button again if you change them. The network parameters that you can change are:
    > __Hidden Layer Size__ - The size of the hidden layer of the network (read above if      you don't know what this means) 
    > __Batch Size__ - The amount of examples used per each batch from which the neural       network trains (read above if you don't know what this means)
    > __Learning Rate__ - How fast the network learns from each batch. This is the number     by which errors are multiplied when calculated from backpropagation (read above if        you don't know what this means) 

3. Either continiously press the 'train-once' button, or toggle the infinite train button. Notice that the number shown in the '% correct in last batch' window starts to go up. Note that you can still adjust the batch size and learning rate while the network is training. 
4. Once you are satisfied with the number shown in the '% correct in last batch' window (reaching around 90% usually takes less than a minute), either stop pressing the 'train-once' button, or untoggle the forever train button. 
5. Alternatively, you can press the 'load preset weights button' on the right side of the screen. These weights were generated after about 3 hours of training, and have around 95% accuracy in classifying digits. 

## USING THE NETWORK TO CLASSIFY YOUR OWN EXAMPLES

This network allows you to classify either 2 or 1 digit numbers. 

##### To classify a 1 digit number:
1. Press the 'setup one digit classification' button
2. Press the 'setup-draw' button
3. Toggle the 'draw' button
4. Draw a 1 digit number on the screen by pressing down your mouse to draw
5. Untoggle the 'draw' button
6. Press the 'classify current drawing (1 digit)' button
7. A window with the networks guess will pop up
8. If you don't feel like drawing a number, you can also load a random digit from the dataset using the corresponding button.

##### To classify a 2 digit number:
1. Press the 'setup two digit classification' button. Note that the screen will get larger.
2. Press the 'setup-draw' button
3. Toggle the 'draw' button
4. Draw a 2 digit number on the screen by pressing down your mouse to draw
5. Untoggle the 'draw' button
6. Press the 'classify current drawing (2 digit)' button
7. A window with the networks guess will pop up
8. Note that 2 digit classification simply consists of attempting to split the number into two parts to classify them separately

## BUGS AND LIMITATIONS

* There will sometimes be an error saying that netlogo has encountered number that are too large for it to handle. There is no known cause or fix to this.
* There will sometimes be errors about netlogo not finding a file in a certain directory. This only happens when you run the program directly from the provided .zip folder (without unzipping it). You can fix this by unzipping the program folder into a non temporary directory.


## CREDITS AND REFERENCES
##### Code credits:
Ivan Galakhov - python (June 2017)
##### Github link:
https://github.com/igalakhov/intro1-FinalProject
##### MNIST dataset:
http://yann.lecun.com/exdb/mnist/
##### Image credits:
http://neuralnetworksanddeeplearning.com/chap1.html
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.0.2
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
