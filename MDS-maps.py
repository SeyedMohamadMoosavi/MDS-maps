#! /usr/bin/env python


#######################################################################################
# This program maps a high dimensional space to a 2D plane using Multidimensional     #
# scalling ifor dimension reduction.                                                  #
# Mohamad Moosavi 15Feb 2018                                                          #
#######################################################################################


import numpy as np
import random
import sys 
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from scipy.stats import kde
from argparse import ArgumentParser
import matplotlib
import json

## Input handler ####################################
parser = ArgumentParser(description="script to generate a MDS plot of a similarity matrix")
parser.add_argument("--similarity-matrix","-sm", dest="SimilarityMatrix",
        help ="file containing the similarity matrix",required=True)
parser.add_argument('--random-seed',"-rnd", dest='RandomSeed', action='store',
        default=1, 
        help='Fixing the random seed to make reproducible results')
parser.add_argument('--similarity-cutoff',"-scut", dest='SimCutoff', action='store',
        default=0.2, 
        help='Cutoff for the connected nodes on the plot')
parser.add_argument('--file-names',"-fn", dest='filenames',action='store',
        help='the file containing the json keys for coloring the map')
parser.add_argument('--coloring-property',"-col", dest='PropertyMatrix', action='store',nargs="*",
        default=None,
        help='Needs two arguments: 1. json file of properties 2. the key to be used for colorcoding the maps')
parser.add_argument('--output-name',"-o", dest='OutputName', action='store',
        default=None, 
        help='The name of figure output file')
parser.add_argument('--prop-colmap', dest='property_colmap', action='store',
        default=None, 
        help='The name of colormap for the representative points')
parser.add_argument('--space-colmap', dest='space_colmap', action='store',
        default=None, 
        help='The name of colormap for the space')
parser.add_argument('--labels', dest='labels', action='store_true',
        default=False, 
        help='labaling dots with array of numbers starting from 0')
parser.add_argument('--labels-file', dest='labels_file', action='store',
        default=False, 
        help='labeling dots with a give name')

args = parser.parse_args()
####################################################

class MDSplot():
    def __init__(self):
        self.SimilarityFile=""
        self.SimilarityCutoff=0.2
        self.RNDseed=1
        self.colmap_lines = plt.cm.get_cmap("Greys")
        self.colmap_property = plt.cm.get_cmap("rainbow")
        self.clf=False
        self.labels=False
        self.labels_file=False
        self.similarity=None
        self.PropMat=None
        self.PropLabels=None
        self.ColorKeys=[]

    def setup(self,options):
        self.SimilarityFile=options.SimilarityMatrix
        with open(options.filenames,"r") as fn:
            self.names=[n.strip() for n in fn.readlines()]
        self.SimilarityCutoff=float(options.SimCutoff)
        self.RNDseed=int(options.RandomSeed)
        np.random.seed(self.RNDseed)
        self.similarity=self.read_tri_matrix(self.SimilarityFile,len(self.names))
        self.labels=options.labels
        self.labels_file=options.labels_file
        if options.space_colmap:
            self.colmap_lines= plt.cm.get_cmap("%s"%options.space_colmap)
        if options.property_colmap:
            self.colmap_property = plt.cm.get_cmap(options.property_colmap)
        if options.OutputName is None:
            try:
                self.outname="MDS_%s"%(self.SimilarityFile.split(".")[:-1][0])
            except IndexError:
                self.outname="MDS_%s"%(self.SimilarityFile)
        else:
            self.outname=options.OutputName

        if options.PropertyMatrix==None:
            print("no coloring..\n")
            self.PropMat={}
            for n in self.names:
                self.PropMat[n]={"arb":1}
            self.ColorKeys=["arb"]

        elif len(options.PropertyMatrix)==1:
            print("Coloring with all keys in the json file\n")
            with open(options.PropertyMatrix[0],"r") as f:  # second argv is the .json file of geometric properties.
                self.PropMat=json.load(f)

            self.ColorKeys=[i for i in self.PropMat[next(iter(self.PropMat.keys()))].keys() if not i.startswith("Structure")]
            self.PropLabels=[i for i in  self.PropMat[next(iter(self.PropMat.keys()))].keys()  if not i.startswith("Structure")]
        else:
            with open(options.PropertyMatrix[0],"r") as f:  # second argv is the .json file of geometric properties.
                self.PropMat=json.load(f)
            self.ColorKeys=[i for i in options.PropertyMatrix[1:]]
            self.PropLabels=[i for i in options.PropertyMatrix[1:]]


    def color_space(self,prop,ind):
        colorparams=[row[ind] for row in prop]
        return colorparams

    def read_tri_matrix(self,filename,n):
        # this function reads a lower triangular matrix and generate the full matrix.
        with open(filename,"r") as inputdata:
            data=inputdata.readlines()
        if not len(data)==n:
            print("the input distance matrix has not the same size of names!\n")
            sys.exit()

        dmat=np.zeros([n,n])
        for i,line in enumerate(data):
            dists=[entry for entry in line.strip().split(",")]
            for j in range(len(dists)):
                if dists[j]=="":
                    dmat[i,j]=0.0
                else:
                    dmat[i,j]=float(dists[j])
                    dmat[j,i]=float(dists[j])

        return dmat


    def compute_MDS_coords(self):
        # Normalize 
        self.similarity = self.similarity/self.similarity.max()
        mds = manifold.MDS(n_components=2, max_iter=30000, eps=1e-12, random_state=self.RNDseed,
                       dissimilarity="precomputed", n_jobs=1)
        pos = mds.fit(self.similarity).embedding_
        nmds = manifold.MDS(n_components=2, metric=False, max_iter=30000, eps=1e-12,
                        dissimilarity="precomputed", random_state=self.RNDseed, n_jobs=1,
                        n_init=1)
        npos = nmds.fit_transform(self.similarity, init=pos)
        mds_coords=npos
    
        if self.clf:
            # Rotate the data
            clf = PCA(n_components=2)
            mds_coords = clf.fit_transform(mds_coords)

        return mds_coords

    def plotMDS(self):
        pos=self.compute_MDS_coords()
        segments=[]
        sims=[]
        n_space=len(self.similarity)
        for i in range(n_space):
            for j in range(n_space):
                if self.similarity[i,j]>0.05 and self.similarity[i,j]<self.SimilarityCutoff:
                    p1=[pos[i,0],pos[i,1]]
                    p2=[pos[j,0],pos[j,1]]
                    #segments.append([pos[i, :], pos[j, :]])
                    segments.append([p1,p2])
                    sims.append(self.similarity[i,j])
        
        sims=np.array(sims)
        ## Plotting starts here ##
        for figid,key in enumerate(self.ColorKeys):
            colors=[self.PropMat[name][key] for name in self.names]
            try:
                colors=[float(v) for v in colors]
                discrete_values=False 
            except ValueError:
                discrete_values=True
                color_classes=[]
                for v in colors:
                    if not v in color_classes:
                        color_classes.append(v)
                colors=[color_classes.index(v) for v in colors]

            if not discrete_values:
               plt.figure(figid)
               f, ax = plt.subplots()
               plt.axis('off')
               if len(segments)>0:
                   lc = LineCollection(segments,
                                       zorder=0, cmap=self.colmap_lines,
                                       norm=plt.Normalize(0,0.5)) 
                   lc.set_array(sims)
                   lc.set_linewidths(0.8 * np.ones(len(segments)))
                   ax.add_collection(lc)
               colors=np.array(colors)
    
               s_map = plt.cm.ScalarMappable(cmap=self.colmap_property)
               s_map.set_array(colors)
               #sc=plt.scatter(pos[:,0], pos[:,1], c=colors,  vmin=np.amin(self.ColMat[:,col]), vmax=np.amax(self.ColMat[:,col]), s=50, lw=0, cmap=self.colmap_property,alpha=1.0)
               sc=plt.scatter(pos[:,0], pos[:,1], c=colors,  vmin=np.amin(colors), vmax=np.amax(colors), s=50, lw=0, cmap=self.colmap_property,alpha=1.0)
               if self.labels:
                   ## adding labels ##
                   for i, coord in enumerate(pos[:]):
                       theta=random.uniform(1, 2.0*np.pi) 
                       lcoord=coord+0.01*(np.array([np.cos(theta),np.sin(theta)]))
                       ax.annotate(i, (lcoord[0],lcoord[1]),fontsize=6)

               if self.labels_file:
                   ## adding labels ##
                   with open(self.labels_file,"r") as fl:
                       labels=[lab.strip() for lab in fl.readlines()]
                   for i, coord in enumerate(pos[:]):
                       theta=random.uniform(1, 2.0*np.pi) 
                       lcoord=coord+0.01*(np.array([np.cos(theta),np.sin(theta)]))
                       ax.annotate(labels[i], (lcoord[0],lcoord[1]),fontsize=6)

               if self.PropLabels:
                   cbaar=plt.colorbar(sc,shrink=0.8,orientation='horizontal',ticks=[np.amin(colors),np.amax(colors)])
                   cbaar.set_label('%s'%(self.PropLabels[figid]),fontsize=14)
                   cbar_label=([np.amin(colors),np.amax(colors)])
                   cbaar.set_ticklabels(cbar_label,update_ticks=True)
               if self.PropLabels:
                   print(key)
                   name_output="_".join([tx for tx in self.PropLabels[figid].strip().split() if not ("[" in tx or "(" in tx)])
                   plt.savefig("%s_%s.png"%(self.outname,name_output),dpi=300)
               else:
                   plt.savefig("%s_%i.png"%(self.outname,figid),dpi=300)
               plt.close('all')

            else:
               fig, ax = plt.subplots()
               plt.axis('off')
               lc = LineCollection(segments,
                                   zorder=0, cmap=self.colmap_lines,
                                   norm=plt.Normalize(0,0.5)) 
               lc.set_array(sims)
               lc.set_linewidths(0.8 * np.ones(len(segments)))
               ax.add_collection(lc)
               normalize = matplotlib.colors.Normalize(vmin=min(colors), vmax=max(colors))
               s_map = plt.cm.ScalarMappable(cmap=self.colmap_property,norm=normalize)
               s_map.set_array(colors)
               for i, coord in enumerate(pos[:]):
                   color = self.colmap_property(normalize(colors[i]))
                   plt.plot(pos[i,0], pos[i,1], c=color,marker="o",markersize=7,mew=0.6,alpha=0.7)#, s=50, lw=0,alpha=1.0)
            
               if self.PropLabels:
                   halfdist = 1/2.0
                   boundaries = np.linspace(min(colors) - halfdist, max(colors) + halfdist, len(set(colors)) + 1)
                   cbaar=plt.colorbar(s_map,shrink=0.8,orientation='horizontal',ticks=boundaries+halfdist, boundaries=boundaries)
                   cbaar.ax.set_xticklabels(boundaries+halfdist,rotation=0)
                   cbaar.set_label('%s'%(self.PropLabels[0]),fontsize=14)
                   cbar_ticks=(color_classes)
                   cbaar.set_ticklabels(cbar_ticks,update_ticks=True)
            
               if self.labels:
                   ## adding labels ##
                   for i, coord in enumerate(pos[:]):
                       theta=random.uniform(1, 2.0*np.pi) 
                       coordp=coord+0.01*(np.array([np.cos(theta),np.sin(theta)]))
                       ax.annotate(i, (coordp[0],coordp[1]),fontsize=6)
            
               if self.PropLabels:
                   plt.savefig("%s_%s.png"%(self.outname,self.PropLabels[0]),dpi=600)
                   plt.savefig("%s_%s.pdf"%(self.outname,self.PropLabels[0]),format="pdf",dpi=600)
               else:
                   plt.savefig("%s_%i.png"%(self.outname,col),dpi=600)
               plt.close('all')




if __name__=="__main__":
    MDS = MDSplot()
    MDS.setup(args)
    MDS.plotMDS()

