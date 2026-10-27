# Background cartoons

`trace.py` turns a line drawing into Fourier epicycle data for `../fourier.js`.
It writes `data/NAME.json` (plus a preview, `data/NAME.png`) and rebuilds `../characters.js`.
`order.json` sets the play order.

Needs `numpy`, `opencv-python-headless` and `scikit-image`. These are the commands used for the current set:

```
python trace.py zoidberg.jpg zoidberg "dr. zoidberg"     --terms 350 --blank 0,0,0.3,0.06 --thresh 190 --close 5
python trace.py homer.png    homer    "homer simpson"    --terms 350 --blank 0.6,0.96,1,1 --size 1000
python trace.py cartman.png  cartman  "eric cartman"     --terms 350 --blank 0.86,0.95,1,1 --size 1200 --hollow 9
python trace.py slurms.jpg   slurms   "slurms mackenzie" --terms 350 --thresh 70 --close 3 --size 900 \
    --blank 0,0,1,0.1 --blank 0,0.755,1,1 --blank 0,0.08,0.5,0.25 --blank 0.9,0.25,1,0.45 --blank 0,0,0.35,1 --blank 0.82,0.5,1,1
python trace.py hutz.png     hutz     "lionel hutz"      --terms 350 --thresh 90 --close 3
python trace.py randy.png    randy    "randy marsh"      --terms 350 --edges 120 --close 5
```

Clean line art (coloring pages) works best with the default dark-ink mode. For coloured renders whose
outlines aren't black (South Park), use `--edges`, which traces the boundaries between colour regions.
Use `--blank` to wipe watermarks and backgrounds. Check `data/NAME.png` after each run.

Sources: Zoidberg from a drawingforall.net tutorial, Homer and Cartman from easydrawingguides.com coloring pages, Hutz from the Simpsons Wiki, Slurms from The Infosphere, and Randy from Wikipedia.
