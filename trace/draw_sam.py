import cv2, numpy as np, math, sys
W,H=700,800
img=np.full((H,W),255,np.uint8)
T=5
def bez(*P,n=60):
    P=np.array(P,float); out=[]
    for t in np.linspace(0,1,n):
        pts=P.copy()
        while len(pts)>1: pts=(1-t)*pts[:-1]+t*pts[1:]
        out.append(pts[0])
    return np.array(out)
def path(segs,closed=False):
    pts=np.vstack(segs).astype(np.int32)
    cv2.polylines(img,[pts],closed,0,T,cv2.LINE_AA)
def curve(*P): path([bez(*P)])

# face outline: temples -> cheeks -> jaw -> chin
path([bez((205,300),(195,400),(205,470),(245,540)),
      bez((245,540),(285,600),(320,625),(350,627)),
      bez((350,627),(380,625),(415,600),(455,540)),
      bez((455,540),(495,470),(505,400),(495,300))])
# ears
curve((207,370),(170,350),(165,430),(215,455))
curve((493,370),(530,350),(535,430),(485,455))
curve((198,395),(188,405),(195,425),(208,428))
# hair: messy curls around top
cx,cy=350,290
rng=np.random.default_rng(3)
angles=np.linspace(math.pi*1.03,math.pi*1.97,0)  # unused
outer=[]
pts=[(200,345),(178,295),(168,240),(180,185),(215,140),(265,108),(325,92),(385,92),(440,108),(488,140),(522,190),(532,248),(522,300),(500,345)]
segs=[]
for (x0,y0),(x1,y1) in zip(pts[:-1],pts[1:]):
    mx,my=(x0+x1)/2,(y0+y1)/2
    dx,dy=x1-x0,y1-y0; L=math.hypot(dx,dy)
    nx,ny=dy/L,-dx/L  # outward normal (clockwise path)
    b=30+rng.uniform(-8,14)
    segs.append(bez((x0,y0),(x0+nx*b,y0+ny*b),(x1+nx*b,y1+ny*b),(x1,y1)))
path(segs)
# hairline / fringe curls across forehead
fr=[(205,300),(225,255),(265,240),(300,215),(345,230),(385,205),(425,228),(465,225),(495,300)]
segs=[]
for (x0,y0),(x1,y1) in zip(fr[:-1],fr[1:]):
    segs.append(bez((x0,y0),((x0+x1)/2-10,min(y0,y1)-35),((x0+x1)/2+15,max(y0,y1)+10),(x1,y1)))
path(segs)
# strands
for P in [((250,200),(270,160),(300,175),(310,135)),((320,190),(340,150),(370,170),(385,125)),
          ((400,200),(425,165),(450,185),(465,145)),((215,250),(225,215),(250,225),(255,190)),
          ((470,250),(490,220),(475,200),(500,180))]:
    curve(*P)
# eyebrows
curve((255,355),(275,335),(310,333),(330,345))
curve((370,345),(392,333),(425,335),(445,355))
# eyes (smiling almond)
curve((262,390),(280,372),(310,372),(325,388))
curve((262,390),(285,400),(308,400),(325,388))
curve((375,388),(390,372),(420,372),(438,390))
curve((375,388),(392,400),(415,400),(438,390))
cv2.circle(img,(294,386),6,0,-1,cv2.LINE_AA)
cv2.circle(img,(406,386),6,0,-1,cv2.LINE_AA)
# nose
curve((345,400),(340,440),(330,465),(338,478))
curve((338,478),(350,488),(362,486),(372,474))
# smile
curve((280,515),(315,560),(385,560),(420,515))
curve((282,512),(286,518),(288,522),(290,526))
curve((418,512),(414,518),(412,522),(410,526))
# neck
curve((305,610),(303,635),(300,655),(292,672))
curve((395,610),(397,635),(400,655),(408,672))
# shirt: shoulders + neckline
curve((292,672),(230,700),(150,720),(110,790))
curve((408,672),(470,700),(550,720),(590,790))
curve((292,672),(310,730),(390,730),(408,672))
curve((296,690),(322,735),(378,735),(404,690))
cv2.imwrite(sys.argv[1],img)
