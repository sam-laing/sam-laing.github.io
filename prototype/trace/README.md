# Background cartoons

`trace.py` turns a line drawing into Fourier epicycle data for `../fourier.js`.
It writes `data/NAME.json` (plus a preview, `data/NAME.png`) and rebuilds `../characters.js`.
`order.json` sets the play order.

Needs `numpy`, `opencv-python-headless` and `scikit-image`. These are the commands used for the current set:

```
python trace.py zoidberg.jpg zoidberg "dr. zoidberg" --blank 0,0,0.3,0.06 --thresh 190 --close 5
python trace.py cartman.png  cartman  "eric cartman" --blank 0.86,0.95,1,1 --size 1200 --hollow 9
python trace.py hutz.png     hutz     "lionel hutz"  --thresh 90 --close 3
```

Sources: Zoidberg from the drawingforall.net tutorial, Cartman from an easydrawingguides.com coloring page, and Hutz from the Simpsons Wiki infobox render.
