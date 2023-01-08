import numpy as np
import matplotlib.pylab as plt
from PIL import Image as image
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import os



#### =====================================================
####                      Input
#### =====================================================



file = os.path.join(Path('__File__').parent, 'datafiles', 'lisboa.jpg')

k_cluster = 4

#### =====================================================
####                      Function
#### =====================================================

# la distance euclidienne
def dist(a,b,ax=1):
    return np.linalg.norm(a-b,axis=1)

#### =====================================================
####                      EXE
#### =====================================================

fig=plt.figure()
ax=Axes3D(fig)


im              = image.open(file,'r')
width, height   = im.size
im_data         = np.array(im.getdata())

# Generate 
x_pos = np.random.randint(0,np.max(im_data),size=k_cluster)
y_pos = np.random.randint(0,np.max(im_data),size=k_cluster)
z_pos = np.random.randint(0,np.max(im_data),size=k_cluster)


XYZ_pos     = np.array(list(zip(x_pos,y_pos,z_pos)))
XYZ_pos_0   = np.zeros(XYZ_pos.shape)
clusters    = np.zeros(len(im_data))
error       = dist(XYZ_pos,XYZ_pos_0)

n=0

while error.all() !=0:

    for i in range(len(im_data)):
        distance=dist(im_data[i],XYZ_pos)
        #print(dist,i)
        cluster=np.argmin(distance)
        clusters[i]=cluster
    XYZ_pos_0=deepcopy(XYZ_pos)

    for i in range(k_cluster):
        points=[im_data[j] for j in range(len(im_data)) if clusters[j]==i]
        if (len(points)>0):
            XYZ_pos[i]=np.mean(points,axis=0)
    error=dist(XYZ_pos,XYZ_pos_0,None)
    print ("Finished iteration "+str(n))
    if(n>16):
        break
    n=n+1

colors=['r','g','b','y','c','m','red','k','yellow']

for i in range(k_cluster):
    points=np.array([im_data[j] for j in range (len(im_data)) if clusters[j]==i])
    if (len(points)>0):
        ax.scatter(points[:,0],points[:,1],points[:,2],c=colors[i])


ax.scatter(XYZ_pos[:,0], XYZ_pos[:,1], XYZ_pos[:,2], marker='*',s=200, c='#050505')
ax.set_xlabel('Red pixel')
ax.set_ylabel('Green pixel')
ax.set_zlabel('Blue pixel')

plt.show()
plt.close()

plt.figure(figsize=(20,10))
plt.imshow(np.reshape(clusters,(width,height)))
plt.colorbar()
plt.show()



