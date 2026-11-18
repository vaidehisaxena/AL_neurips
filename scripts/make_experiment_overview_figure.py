#!/usr/bin/env python3
"""Create a compact graphical workflow for the agentic AL experiment."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "technical_report"
C = {
    "phlox": "#CAA9F3", "verbena": "#B37AD4", "periwinkle": "#7997E6",
    "atlantis": "#206ABC", "phthalo": "#0E155E", "ink": "#14172B",
    "muted": "#626A7D", "line": "#D9DEEB", "paper": "#FFFFFF",
}


def tint(color: str, amount: float = 0.86):
    rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(rgb * (1 - amount) + amount)


def arrow(ax, a, b, *, rad=0, lw=2.2, color=None, ms=16):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color or C["phthalo"], connectionstyle=f"arc3,rad={rad}",
        shrinkA=3, shrinkB=3, zorder=8,
    ))


def node(ax, x, y, w, h, number, label, accent):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.16",
        facecolor=tint(C[accent], 0.91), edgecolor=C[accent], linewidth=2, zorder=2,
    ))
    ax.add_patch(Circle((x + 0.27, y + h - 0.28), 0.18, color=C[accent], zorder=4))
    ax.text(x + 0.27, y + h - 0.28, str(number), ha="center", va="center",
            color="white", fontsize=10, weight="bold", zorder=5)
    ax.text(x + 0.52, y + h - 0.28, label, ha="left", va="center",
            color=C["phthalo"], fontsize=11, weight="bold", zorder=5)


def molecules(ax, x, y):
    pts = [(x-.55,y+.22),(x-.18,y+.42),(x+.18,y+.23),(x+.55,y+.40),
           (x-.52,y-.22),(x-.10,y-.12),(x+.32,y-.18),(x+.62,y-.34)]
    cols = ["phlox","periwinkle","verbena","atlantis","periwinkle","phthalo","phlox","atlantis"]
    for i, j in [(0,1),(1,2),(2,3),(0,4),(1,5),(2,6),(3,7),(4,5),(5,6),(6,7)]:
        ax.plot([pts[i][0],pts[j][0]],[pts[i][1],pts[j][1]],color=C["phthalo"],lw=1.4,alpha=.55,zorder=3)
    for (px,py), col in zip(pts, cols):
        ax.add_patch(Circle((px,py),.105,facecolor=C[col],edgecolor="white",linewidth=1.4,zorder=5))


def gp_plot(ax, x, y):
    xx = np.linspace(-.68, .68, 80)
    yy = .10 + .24*np.sin(3*xx) + .13*xx
    band = .15 + .12*np.abs(xx)
    ax.fill_between(x+xx,y+yy-band,y+yy+band,color=C["periwinkle"],alpha=.28,zorder=3)
    ax.plot(x+xx,y+yy,color=C["atlantis"],lw=2.6,zorder=4)
    obsx=np.array([-.56,-.27,.04,.35,.61])
    obsy=.10+.24*np.sin(3*obsx)+.13*obsx+np.array([.03,-.04,.02,-.02,.03])
    ax.scatter(x+obsx,y+obsy,s=30,color=C["phthalo"],edgecolor="white",linewidth=.9,zorder=5)
    ax.text(x,y-.54,r"$\mu(x)\;\pm\;\sigma(x)$",ha="center",va="center",fontsize=10,color=C["muted"])


def controller(ax, x, y):
    ax.add_patch(FancyBboxPatch((x-.62,y+.02),1.24,.60,boxstyle="round,pad=.02,rounding_size=.12",
                                facecolor=C["verbena"],edgecolor="none",zorder=3))
    ax.add_patch(Polygon([[x-.13,y+.02],[x+.02,y-.12],[x+.14,y+.02]],closed=True,color=C["verbena"],zorder=3))
    ax.text(x,y+.32,"LLM",ha="center",va="center",color="white",fontsize=16,weight="bold",zorder=4)
    left,width,by=x-.67,1.34,y-.42
    parts=[("E",.50,"phthalo"),("U",.28,"periwinkle"),("D",.22,"phlox")]
    cur=left
    for lab,frac,col in parts:
        ax.add_patch(Rectangle((cur,by),width*frac,.24,facecolor=C[col],edgecolor="white",linewidth=1,zorder=4))
        ax.text(cur+width*frac/2,by+.12,lab,ha="center",va="center",fontsize=8.5,
                color="white" if col=="phthalo" else C["phthalo"],weight="bold",zorder=5)
        cur += width*frac
    ax.text(x,by-.15,"30",ha="center",va="top",fontsize=10,weight="bold",color=C["phthalo"])


def queue(ax, x, y):
    colors=[C["phthalo"],C["periwinkle"],C["phlox"]]
    for row,(col,lab) in enumerate(zip(colors,["E","U","D"])):
        yy=y+.42-row*.36
        ax.text(x-.58,yy,lab,ha="center",va="center",fontsize=9,weight="bold",color=col)
        for i in range(5):
            ax.add_patch(Circle((x-.36+i*.23,yy),.13-i*.011,facecolor=col,alpha=1-i*.11,
                                edgecolor="white",linewidth=.8,zorder=4))
    ax.text(x,y-.55,"rank + deduplicate",ha="center",va="center",fontsize=9.3,color=C["muted"])


def assay(ax, x, y):
    vals=np.array([0,1,2,1,3,2,1,3,4,2,1,0,2,4,3,1,2,4,0,2,1,4,3,2,1,3,2,4,1,3])
    palette=[tint(C["phlox"],.6),C["phlox"],C["verbena"],C["periwinkle"],C["phthalo"]]
    for r in range(5):
        for c in range(6):
            ax.add_patch(Circle((x-.57+c*.23,y+.46-r*.23),.082,
                                facecolor=palette[vals[r*6+c]],edgecolor="white",linewidth=.8,zorder=4))
    ax.text(x,y-.55,"reveal labels",ha="center",va="center",fontsize=9.3,color=C["muted"])


def mini_card(ax, x, y, w, h, title, stat, accent, kind):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.018,rounding_size=.13",
                                facecolor="white",edgecolor=C["line"],linewidth=1.4,zorder=2))
    ax.add_patch(Rectangle((x,y),.09,h,facecolor=C[accent],edgecolor="none",zorder=3))
    ax.text(x+.25,y+h-.25,title,ha="left",va="top",fontsize=10.5,weight="bold",color=C["phthalo"])
    ax.text(x+w-.20,y+h-.25,stat,ha="right",va="top",fontsize=9.4,weight="bold",color=C[accent])
    cx,cy=x+w/2,y+.56
    if kind=="bars":
        for i,(hh,col,lab) in enumerate(zip([.35,.68,.95],["phlox","periwinkle","phthalo"],["R","F","A"])):
            xx=cx-.55+i*.55
            ax.add_patch(Rectangle((xx,cy-.28),.28,hh,facecolor=C[col],edgecolor="none",zorder=4))
            ax.text(xx+.14,cy-.38,lab,ha="center",va="top",fontsize=8.5,color=C["muted"])
    elif kind=="check":
        ax.add_patch(Circle((cx,cy+.15),.52,facecolor=tint(C[accent],.78),edgecolor=C[accent],linewidth=2,zorder=3))
        ax.plot([cx-.26,cx-.04,cx+.32],[cy+.12,cy-.10,cy+.40],color=C["phthalo"],lw=4,solid_capstyle="round",zorder=5)
        ax.text(cx,cy-.52,"next batch",ha="center",va="top",fontsize=8.8,color=C["muted"])
    elif kind=="ablate":
        xs=np.linspace(cx-.72,cx+.72,5)
        for i,(xx,hh) in enumerate(zip(xs,[.83,.77,.80,.79,.83])):
            col=["phthalo","periwinkle","phlox","verbena","atlantis"][i]
            ax.add_patch(Rectangle((xx-.09,cy-.28),.18,hh,facecolor=C[col],edgecolor="none",zorder=4))
        ax.plot([cx-.82,cx+.82],[cy+.55,cy+.55],color=C["muted"],lw=1,ls="--",zorder=3)
    elif kind=="availability":
        for r in range(4):
            for c in range(5):
                on=(r*5+c)<16
                ax.add_patch(Circle((cx-.55+c*.28,cy+.48-r*.28),.09,
                                    facecolor=C[accent] if on else C["line"],edgecolor="white",linewidth=.6,zorder=4))
        ax.text(cx,cy-.52,"80% accessible",ha="center",va="top",fontsize=8.8,color=C["muted"])


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(15.4,8.5),facecolor=C["paper"])
    ax.set_xlim(0,15.4); ax.set_ylim(0,8.5); ax.axis("off")
    ax.text(14.9,8.11,"4 targets  |  10 seeds",fontsize=10.5,weight="bold",color=C["muted"],ha="right",va="top")

    w,h,gap,top_y=2.55,2.18,.40,5.18
    xs=[.48+i*(w+gap) for i in range(5)]
    for i,(label,accent) in enumerate(zip(["Start","Model","Decide","Select","Assay"],
                                          ["phlox","periwinkle","verbena","atlantis","phthalo"])):
        node(ax,xs[i],top_y,w,h,i+1,label,accent)
    molecules(ax,xs[0]+w/2,top_y+.86)
    ax.text(xs[0]+w/2,top_y+.18,"60 observed",ha="center",va="center",fontsize=10,weight="bold",color=C["phthalo"])
    gp_plot(ax,xs[1]+w/2,top_y+.83)
    controller(ax,xs[2]+w/2,top_y+.79)
    queue(ax,xs[3]+w/2,top_y+.88)
    assay(ax,xs[4]+w/2,top_y+.90)
    for i in range(4): arrow(ax,(xs[i]+w+.03,top_y+h/2),(xs[i+1]-.03,top_y+h/2))

    # Feedback travels below the process row, keeping the five stages visually clean.
    assay_x=xs[4]+w/2
    model_x=xs[1]+w/2
    loop_y=4.56
    ax.plot([assay_x,assay_x,model_x],[top_y-.03,loop_y,loop_y],
            color=C["phthalo"],lw=2.2,zorder=6)
    arrow(ax,(model_x,loop_y),(model_x,top_y-.03),lw=2.2)
    ax.text((assay_x+model_x)/2,4.72,"x10 cycles",ha="center",va="center",fontsize=12,weight="bold",color=C["phthalo"])
    ax.text((assay_x+model_x)/2,4.43,"30 compounds / cycle",ha="center",va="center",fontsize=9.6,color=C["muted"])

    endpoint_x=14.42
    endpoint_y=4.45
    ax.add_patch(Circle((endpoint_x,endpoint_y),.43,facecolor=C["phthalo"],edgecolor="none",zorder=7))
    ax.text(endpoint_x,endpoint_y+.06,"360",ha="center",va="center",fontsize=14,weight="bold",color="white",zorder=8)
    ax.text(endpoint_x,endpoint_y-.16,"measured",ha="center",va="center",fontsize=8.3,color="white",zorder=8)
    arrow(ax,(xs[4]+w*.82,top_y-.02),(endpoint_x,endpoint_y+.42),rad=0,lw=1.8)

    ay,aw,ah,ag=.72,3.45,2.63,.24
    axs=[.5+i*(aw+ag) for i in range(4)]
    mini_card(ax,axs[0],ay,aw,ah,"Benchmark","40 + 360 runs","phlox","bars")
    mini_card(ax,axs[1],ay,aw,ah,"Forecasts","686 claims","verbena","check")
    mini_card(ax,axs[2],ay,aw,ah,"Ablations","200 runs","periwinkle","ablate")
    mini_card(ax,axs[3],ay,aw,ah,"Availability","80% pool","atlantis","availability")

    ax.text(.55,.34,"READOUTS",fontsize=8.8,weight="bold",color=C["muted"],va="center")
    mx=1.65
    for label,col,step in [("Top-k recall","phlox",2.75),("Regret","verbena",2.75),
                           ("Scaffolds","periwinkle",2.75),("Forecast success","atlantis",3.3)]:
        ax.add_patch(Circle((mx,.34),.07,facecolor=C[col],edgecolor="none"))
        ax.text(mx+.13,.34,label,fontsize=9.3,color=C["ink"],va="center")
        mx += step

    fig.tight_layout(pad=0)
    for suffix in ("png","svg","pdf"):
        kwargs={"bbox_inches":"tight","facecolor":C["paper"]}
        if suffix=="png": kwargs["dpi"]=320
        fig.savefig(OUT/f"experiment_overview.{suffix}",**kwargs)
    plt.close(fig)


if __name__=="__main__":
    build()
