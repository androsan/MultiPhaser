import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
import cv2
from scipy import misc
from Tresholding import FindTreshold
import csv
import time
import datetime
from argparse import Namespace

from skimage.exposure import cumulative_distribution

def cdf(im):
 '''
 computes the CDF of an image im as 2D numpy ndarray
 "'''
 c, b = cumulative_distribution(im) 
 # pad the beginning and ending pixels and their CDF values
 c = np.insert(c, 0, [0]*b[0])
 c = np.append(c, [1]*(255-b[-1]))
 return c

def hist_matching(c, c_t, im):
 '''
 c: CDF of input image computed with the function cdf()
 c_t: CDF of template image computed with the function cdf()
 im: input image as 2D numpy ndarray
 returns the modified pixel values
 ''' 
 pixels = np.arange(256)
 # find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of   
 # the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
 new_pixels = np.interp(c, c_t, pixels) 
 im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
 return im

# CV2 Gamma Correction Function
def Gamma_Correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    GC = cv2.LUT(image, table)
    return GC


# HITRO SEKCIONIRANJE SLIKE

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

modra = (51,102,255)   
zelena = (155,255,0)
purple = (102, 0, 255)
yellow = (255, 255, 102)

def delitelji(x):
    delitlji = []
    for i in range(1,x):
        if x%i==0:
            delitlji.append(i)
    return delitlji

def fast_sections(mapa, fajl, X_sections, Y_sections, color_to_count):
        global list_of_cut_thresholds, num_vrstic, num_kolon, mreza, OCV_podatki
        if mapa is None:
            display1.config(fg='red')
            ime_slike.set('..najprej izberi sliko!')
        else:
            file = mapa+'/'+fajl
            irfan = cv2.imread(file,0)
            Xcut = [padX1.get()/100, padX2.get()/100]; Ycut = [padY1.get()/100, padY2.get()/100]
            img = irfan[int(irfan.shape[0]*Ycut[0]): int(irfan.shape[0]*Ycut[1]), int(irfan.shape[1]*Xcut[0]): int(irfan.shape[1]*Xcut[1])]
            if CB_var.get()==1:
                img = cv2.convertScaleAbs(img, alpha=brightness_var.get(), beta=contrast_var.get())
                img = Gamma_Correction(img, gamma=gamma_var.get())
            Y=img.shape[0]; X=img.shape[1]
            del_x=delitelji(X)
            x_closest = [abs(X_sections - i) for i in del_x]
            s=x_closest.index(min(x_closest))
            kol = del_x[s]
            kolon = int(X/kol)
            del_y=delitelji(Y)
            y_closest = [abs(Y_sections - i) for i in del_y]
            š=y_closest.index(min(y_closest))
            vrs = del_y[š]
            vrstic = int(Y/vrs)
            list_of_cut_thresholds=[] 
            div=blockshaped(img, vrstic, kolon)
            num_vrstic = int(Y/vrstic)
            num_kolon = int(X/kolon)       
            if cut_mode.get()== 'thresh':
                for sec in div:
                    v = FindTreshold(sec)
                    list_of_cut_thresholds.append(v[0])
            elif cut_mode.get()== 'otsu':
                for sec in div:
                    ret, otsu = cv2.threshold(sec,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    list_of_cut_thresholds.append(ret)

            elif cut_mode.get()=='opencv':
                OCV_podatki = []
                for sec in div:
                    Z = sec.reshape((-1,3))
                    Z = np.float32(Z)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    K = 3
                    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                    center[0]=[255,255,255]    # matrica (bela barva)
                    center[1]=[0,0,0]              # vse sekundarne faze (črna barva)
                    center = np.uint8(center)
                    res = center[label.flatten()]
                    res2 = res.reshape((sec.shape))
                    elf = np.stack((res2,res2,res2),axis=2)
                    b=(label==1).sum()
                    bb=(label==2).sum()
                    vsi = label.shape[0]
                    OCV_podatki.append((elf, vsi, b, bb))

                w=[i[0] for i in OCV_podatki]
                ww=np.asarray(w)
                www=np.reshape(ww, (num_vrstic, num_kolon, ww.shape[1], ww.shape[2], ww.shape[3]))
                wrste=[]
                for i in www:
                    h = np.hstack( (np.asarray( j ) for j in i))
                    wrste.append(h)
                www_zlep = np.vstack( (np.asarray( i ) for i in wrste ) )
                plt.figure()
                plt.imshow(www_zlep)

            qqq_zlep, mesh_total, mesh_grey, mesh_black, CB= None, None, None, None, None
            if mesh_var.get()== 'mesh_ON':
                kol=[]; cutties = []
                ft = min(list_of_cut_thresholds)
                for sec, gt in zip(div, list_of_cut_thresholds):
                        vsi = (sec<=255).sum()
                        b = (sec<ft).sum()
                        s = (sec<gt).sum()
                        black = round(100*b/vsi, 2)
                        grey = round (100*(s-b)/vsi, 2)
                        total = round(black+grey, 1)           
                        im_black = np.stack((sec, sec, sec), axis=2)
                        im_grey = np.stack((sec, sec, sec), axis=2)
                        B=np.full((int(b),3), modra).astype('int')
                        S=np.full((int(s),3), zelena).astype('int')
                        try:
                            BC=np.concatenate(B, axis=0)
                            im_black[im_black<ft]=BC
                            del BC
                        except ValueError:
                            pass
                        try:
                            SC=np.concatenate(S, axis=0)
                            im_grey[ im_grey<gt]=SC
                            del SC, S
                        except ValueError:
                            pass                
                        try:
                            im_grey[np.all(im_black==modra, axis=2)]=B
                            del B
                        except UnboundLocalError:
                            pass

                        kol.append((im_grey, black, grey, total))
                        ct = np.stack((sec, sec, sec), axis=2)
                        ct[0]=color_to_count; ct[1]=color_to_count
                        ct[-1]=color_to_count; ct[-2]=color_to_count
                        ct[:,0]=color_to_count; ct[:,1]=color_to_count
                        ct[:,-1]=color_to_count; ct[:,-2]=color_to_count
                        cutties.append(ct)

                t=[i[-1] for i in kol]; tt=[i[-2] for i in kol]; ttt=[i[-3] for i in kol] 
                #plt.plot(t)
                mesh_total=round(sum(t)/len(t), 1)
                mesh_grey=round(sum(tt)/len(tt), 1)
                mesh_black=round(sum(ttt)/len(ttt), 1)
                q=[i[0] for i in kol]
                qq=np.asarray(q)
                qqq=np.reshape(qq, (num_vrstic, num_kolon, qq.shape[1], qq.shape[2], qq.shape[3]))
                vrste=[]
                for i in qqq:
                    h = np.hstack( (np.asarray( j ) for j in i))
                    vrste.append(h)
                qqq_zlep = np.vstack( (np.asarray( i ) for i in vrste ) )
                """
                if SM.get()=='closed':
                    plt.figure()
                    plt.title(cut_mode.get()+', zlepljeno: '+str(mesh_total)+' % sekundarnih faz')
                    plt.imshow(qqq_zlep)
                """
                oo = np.asarray(cutties)
                ooo = np.reshape(oo, (num_vrstic, num_kolon, oo.shape[1], oo.shape[2], oo.shape[3]))
                mesh=[]
                for i in ooo:
                    h = np.hstack( (np.asarray( j ) for j in i))
                    mesh.append(h)
                mreza = np.vstack( (np.asarray( i ) for i in mesh ) )
                CB = ('raw', 'raw', 'raw')
                if CB_var.get()==1:
                    CB=CB_function()
            return list_of_cut_thresholds, num_vrstic, num_kolon, qqq_zlep, mesh_total, mesh_grey, mesh_black, CB


def Show_Mesh(mapa, fajl, X_sections, Y_sections, color_to_count):
    SM.set('open')
    fast_sections(mapa, fajl, X_sections, Y_sections, color_to_count)
    if mapa:
        plt.figure()
        plt.title('Mreža '+str(num_kolon)+' x '+str(num_vrstic))
        plt.imshow(mreza)
        SM.set('closed')
    else:
        SM.set('closed')
    
plt.ion()

def global_threshold(mapa, fajl):
    file = mapa+'/'+fajl
    irfan = cv2.imread(file,0)
    Xcut = [padX1.get()/100, padX2.get()/100]; Ycut = [padY1.get()/100, padY2.get()/100]
    img = irfan[int(irfan.shape[0]*Ycut[0]): int(irfan.shape[0]*Ycut[1]), int(irfan.shape[1]*Xcut[0]): int(irfan.shape[1]*Xcut[1])]
    if CB_var.get()==1:
        img = cv2.convertScaleAbs(img, alpha=brightness_var.get(), beta=contrast_var.get())
        img = Gamma_Correction(img, gamma=gamma_var.get())
    GT = FindTreshold(img)
    minimalc = global_minimum.get()
    aaa = np.histogram(img.flatten(),256,[0,256])
    glm = np.where(aaa[0]==aaa[0].max())
    if cut_mode.get()=='thresh':
        if showhist_var.get()=='hist_ON':
            plt.figure()
            aaa = plt.hist(img.flatten(),256,[0,256], color = 'r')
            plt.title('Global Thresh: '+str(GT[0])+'   MAX bin height at: '+str(glm[0][0])+'    MIN(manual):  '+str(minimalc))
            plt.plot([GT[0], GT[0]], [0, aaa[0].max()])
            plt.plot([glm[0][0], glm[0][0]], [0, aaa[0].max()])
            plt.plot([minimalc, minimalc], [0, aaa[0].max()])
        return minimalc, GT[0]
    elif cut_mode.get()=='otsu':
        if showhist_var.get()=='hist_ON':
            plt.figure()
            aaa = plt.hist(img.flatten(),256,[0,256], color = 'r')
            plt.title('Global Otsu: '+str(GT[1])+'   MAX bin height at: '+str(glm[0][0])+'    MIN(manual):  '+str(minimalc))
            plt.plot([GT[1], GT[1]], [0, aaa[0].max()])
            plt.plot([glm[0][0], glm[0][0]], [0, aaa[0].max()])
            plt.plot([minimalc, minimalc], [0, aaa[0].max()])
        return minimalc, GT[1]


def show_global_threshold(mapa,fajl):
    file = mapa+'/'+fajl
    irfan = cv2.imread(file,0)
    Xcut = [padX1.get()/100, padX2.get()/100]; Ycut = [padY1.get()/100, padY2.get()/100]
    img = irfan[int(irfan.shape[0]*Ycut[0]): int(irfan.shape[0]*Ycut[1]), int(irfan.shape[1]*Xcut[0]): int(irfan.shape[1]*Xcut[1])]
    if CB_var.get()==1:
        img = cv2.convertScaleAbs(img, alpha=brightness_var.get(), beta=contrast_var.get())
        img = Gamma_Correction(img, gamma=gamma_var.get())
    GT = FindTreshold(img)
    minimalc = global_minimum.get()
    aaa = plt.hist(img.flatten(),256,[0,256],  color = 'r')
    glm = np.where(aaa[0]==aaa[0].max())
    if cut_mode.get()=='thresh':
        plt.title('Global Thresh: '+str(GT[0])+'   MAX bin height at: '+str(glm[0][0])+'    MIN(manual):  '+str(minimalc))
        plt.plot([GT[0], GT[0]], [0, aaa[0].max()])
        plt.plot([glm[0][0], glm[0][0]], [0, aaa[0].max()])
        plt.plot([minimalc, minimalc], [0, aaa[0].max()])
        return GT[0]
    elif cut_mode.get()=='otsu':
        plt.title('Global Otsu: '+str(GT[1])+'   MAX bin height at: '+str(glm[0][0])+'    MIN(manual):  '+str(minimalc))
        plt.plot([GT[1], GT[1]], [0, aaa[0].max()])
        plt.plot([glm[0][0], glm[0][0]], [0, aaa[0].max()])
        plt.plot([minimalc, minimalc], [0, aaa[0].max()])
        return GT[1]
    

STD=None
def SUPER_TRESH(threshlist, nrows, ncols):
    global STD
    x=np.arange(255)
    a,b = np.histogram(threshlist,x, density=False)
    #a,b, _ = plt.hist(threshlist,x, density=False); plt.close()
    """
    epsilon = 10
    sp = np.where(a == a.max())[0][0]- epsilon
    zg = np.where(a == a.max())[0][0]+ epsilon
    y=[i for i in threshlist if i > sp and i < zg]
    mu, sigma = norm.fit(y)
    n,m, f = plt.hist(y,x, density=True)#; plt.close()
    gauss = mlab.normpdf( m, mu, sigma)
    #l = plt.plot(m, gauss, 'r--', linewidth=2)
    """
    lowest = np.where(a!=0)[0].min()
    bin_max = np.where(a == a.max())
    highest = bin_max[0][0]
    brightest = np.where(a!=0)[0].max()
    var = inter_var.get()
    inter = highest + var * (brightest-highest)
    #bin_max = np.where(gauss == gauss.max())
    #print(bin_max[0][0])
    if showhist_var.get()=='hist_ON':
        plt.figure()
        a,b, _ = plt.hist(threshlist,x, density=False)
        plt.title('Histogram of thresholds, '+cut_mode.get()+'_cut ('+str(int(nrows))+' x '+str(int(ncols))+')\n'+
        'MIN: '+str(lowest)+'  MAX: '+str(highest)+'  INTER: '+str(inter))
        plt.plot([highest, highest], [0,a.max()])
        plt.plot([lowest, lowest], [0,a.max()])
        plt.plot([inter, inter], [0,a.max()])
    STD =[]
    STD.append((threshlist, x, nrows, ncols, lowest, highest, inter))
    return lowest, highest, inter


def Show_Histogram(mapa, fajl, X_sections, Y_sections, color_to_count):
        if choice_mode.get()=='lokalno':
            if mapa and fajl and X_sections and Y_sections:
                cut = fast_sections(mapa, fajl, X_sections, Y_sections, color_to_count)
                gold = SUPER_TRESH(cut[0], cut[1], cut[2])
                if showhist_var.get()=='hist_OFF':
                    plt.figure()
                    a,b, _ = plt.hist(STD[0][0], STD[0][1], density=False)
                    plt.title('Histogram of thresholds of image sections ('+str(int(STD[0][2]))+' x '+str(int(STD[0][3]))+')\n'+
                    cut_mode.get().upper()+'    MIN: '+str(STD[0][4])+'  MAX: '+str(STD[0][5])+'  INTER: '+str(STD[0][6]))
                    plt.plot([STD[0][4], STD[0][4]], [0,a.max()])
                    plt.plot([STD[0][5], STD[0][5]], [0,a.max()])
                    plt.plot([STD[0][6], STD[0][6]], [0,a.max()])
            else:
                display1.config(fg='red')
                ime_slike.set('..najprej izberi sliko!')
        elif choice_mode.get()=='globalno' and cut_mode.get()!='opencv':
            if mapa:
                show_global_threshold(mapa,fajl)
            else:
                display1.config(fg='red')
                ime_slike.set('..najprej izberi sliko!')
  
         
# Execute button: HISTOGRAM  
def Cut_histogram(mapa, fajl, X_sections, Y_sections, color_to_count):
    if choice_mode.get()=='lokalno':
        cut = fast_sections(mapa, fajl, X_sections, Y_sections, color_to_count)
        gold = SUPER_TRESH(cut[0], cut[1], cut[2])
        return gold
    elif choice_mode.get()=='globalno' and cut_mode.get()!='opencv':
        silver = global_threshold(mapa, fajl)
        return silver[0], silver[1]

podatki = [[None, None, None, None, None, None]]
def change(mapa, fajl, ft, ret):
        global podatki
        if mapa:
            file = mapa+'/'+fajl
        else:
            display1.config(fg='red')
            ime_slike.set('..najprej izberi sliko!')
        irfan = cv2.imread(file,0)
        Xcut = [padX1.get()/100, padX2.get()/100]; Ycut = [padY1.get()/100, padY2.get()/100]
        img = irfan[int(irfan.shape[0]*Ycut[0]): int(irfan.shape[0]*Ycut[1]), int(irfan.shape[1]*Xcut[0]): int(irfan.shape[1]*Xcut[1])]
        CB = ('raw', 'raw', 'raw')
        if CB_var.get()==1:
            img = cv2.convertScaleAbs(img, alpha=brightness_var.get(), beta=contrast_var.get())
            img = Gamma_Correction(img, gamma=gamma_var.get())
            CB=CB_function()
        
        """ MAGIC lines -----------------------------------------------------------------------------------"""
        ft=int(ft); ret=int(ret)
        vsi = (img<=255).sum()
        b = (img<ft).sum()
        s = (img<ret).sum()
        black = round(100*b/vsi, 1)
        grey = round (100*(s-b)/vsi, 1)
        total = round(black+grey, 1)
        im_black = np.stack((img, img, img), axis=2)
        im_grey = np.stack((img, img, img), axis=2)
        del img
        B=np.full((int(b),3), modra).astype('int')
        S=np.full((int(s),3), zelena).astype('int')
        BC=np.concatenate(B, axis=0)
        SC=np.concatenate(S, axis=0)
        im_black[im_black<ft]=BC
        im_grey[ im_grey<ret]=SC
        del S,BC,SC
        im_grey[np.all(im_black==modra, axis=2)]=B
        del B
        """----------------------------------------------------------------------------- end of MAGIC LINES""" 
        black_result.set(str(black)+' %  @  '+str(ft))
        grey_result.set(str(grey)+' %  @  '+str(ret))
        total_result.set(str(total)+' %')
        podatki=[]
        podatki.append((black,grey,total,im_grey, ft, ret))
        return black, grey, total, im_grey, ft, ret, CB


def show_analyzed():
    if nacin.get()== 'auto' and cut_mode.get()=='opencv':
        plt.figure()
        plt.title('TOTAL OpenCV:  '+str(out_opencv[2])+' %')
        plt.imshow(out_opencv[3])   
    elif nacin.get()== 'auto' and cut_mode.get()!= 'opencv' and mesh_var.get()=='mesh_OFF':
        plt.figure()
        plt.title('TOTAL '+cut_mode.get()+'_'+choice_mode.get()+':  '+str(out[2])+' %   @ MIN / MAX threshold: '+str(out[4])+' / '+str(out[5]))
        plt.imshow(out[3])
    elif nacin.get()== 'auto' and cut_mode.get()!= 'opencv' and mesh_var.get()=='mesh_ON':
        plt.figure()
        plt.title(cut_mode.get()+', zlepljeno: '+str(out[2])+' % sekundarnih faz')
        plt.imshow(out[3])
    elif nacin.get()== 'rocno':
        plt.figure()
        plt.title('TOTAL '+nacin.get()+':  '+str(out[2])+' %   @ MIN / MAX threshold: '+str(out[4])+' / '+str(out[5]))
        plt.imshow(out[3]) 
    else:
        display1.config(fg='red')
        ime_slike.set('..podatki niso na voljo!')
    

# OpenCV Machine Learning Kmeans Approach
alfi = [[None, None, None, None, None, None]]
def OpenCV(mapa, fajl, var1):
    global alfi
    if mapa:
        file = mapa+'/'+fajl
    else:
        display1.config(fg='red')
        ime_slike.set('..najprej izberi sliko!')
    irfan = cv2.imread(file,0)
    Xcut = [padX1.get()/100, padX2.get()/100]; Ycut = [padY1.get()/100, padY2.get()/100]
    img = irfan[int(irfan.shape[0]*Ycut[0]): int(irfan.shape[0]*Ycut[1]), int(irfan.shape[1]*Xcut[0]): int(irfan.shape[1]*Xcut[1])]
    CB = ('raw', 'raw', 'raw')
    if CB_var.get()==1:
        CB=CB_function()
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, var1, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center[0]=[255,255,255]    # matrica (bela barva)
    center[1]=[0,0,0]              # vse sekundarne faze (črna barva)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    elf = np.stack((res2,res2,res2),axis=2)
    b=(label==1).sum()
    bb=(label==2).sum()
    vsi = label.shape[0]
    siva = round(100*b/vsi, 2)
    ocv_res = []
    ocv_res.append(siva)
    ocv_res.append(100-siva)
    result_ocv = round(min(ocv_res), 1)  
    black= ' - '; black_result.set(black)
    grey= ' - '; grey_result.set(grey)
    total_result.set(str(result_ocv)+' %')
    ft= ' - '; ret= ' - '
    alfi=[]
    alfi.append((black, grey, result_ocv, elf, ft, ret))
    return black, grey, result_ocv, elf, ft, ret, CB


from matplotlib.ticker import MaxNLocator
def thresh_from_percent(mapa, fajl, X_sections, Y_sections, color_to_count):
    global a,f
    boom = Cut_histogram(mapa, fajl, X_sections, Y_sections, color_to_count)
    a=range(boom[1]-10, boom[1]+10)
    f=[change(mapa, fajl, boom[0], i)[2] for i in a]
    #f_closest = [abs(percent - i) for i in f]
    #su = f_closest.index(min(f_closest))
    ax=plt.figure().gca()
    plt.title('Delež sekundarnih faz v odvisnosti od Threshold-a')
    plt.plot(a, f, color='blue'); plt.scatter(a,f, color='aqua')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Threshold', size=18)
    plt.ylabel('Delež sekundarnih faz (%)', size=18)
    #return a[su]


def CB_function():
    Svetloba = brightness_var.get()
    Kontrast = contrast_var.get()
    Gama = gamma_var.get()
    if Svetloba==1.0 and Kontrast==0 and Gama==1.0:
        CB = ('raw', 'raw', 'raw') 
    elif Svetloba==1.0 and Gama==1.0:
        CB = (str(Kontrast), 'raw', 'raw')
    elif Kontrast==0 and Gama==1.0:
        CB = ('raw', str(Svetloba), 'raw')
    elif Kontrast==0 and Svetloba==1.0:
        CB = ('raw', 'raw', str(Gama))
    elif Gama==1.0:
        CB = (str(Kontrast), str(Svetloba), 'raw')
    elif Kontrast==0:
        CB = ('raw' , str(Svetloba), str(Gama))
    elif Svetloba==1.0:
        CB = (str(Kontrast), 'raw' , str(Gama))  
    else:
        CB = (str(Kontrast), str(Svetloba), str(Gama))
    return CB


#------------ -----------------        ------------------        ----------- EXECUTION line ---   --------   ------

def BIG_RUN(mapa, fajl, X_sections, Y_sections, color_to_count):
    global out, out_opencv
    if nacin.get()== 'auto':
        if cut_mode.get()=='opencv':
            out_opencv=OpenCV(mapa, fajl, open_cv_var1.get())
            save_ext.set('OpenCV_'+str(open_cv_var1.get())+'_')
            if final_var.get()=='final_ON':
                show_analyzed()
            if autosave_var.get()== 'autosave_on':
                save(mapa, fajl, save_ext.get())
            return out_opencv
        elif cut_mode.get()!='opencv':
            if choice_mode.get()=='lokalno' and mesh_var.get()=='mesh_OFF':
                boom = Cut_histogram(mapa, fajl, X_sections, Y_sections, color_to_count)
                out = change(mapa, fajl, boom[0], boom[2])
                save_ext.set(cut_mode.get()+'_cut('+str(num_kolon)+'x'+str(num_vrstic)+')_')
                if final_var.get()=='final_ON':
                    show_analyzed()
                if autosave_var.get()== 'autosave_on':
                    save(mapa, fajl, save_ext.get())
                return out
            elif choice_mode.get()=='lokalno' and mesh_var.get()=='mesh_ON':
                cut = fast_sections(mapa, fajl, X_sections, Y_sections, color_to_count)
                out = cut[6], cut[5], cut[4], cut[3], '-', '-', cut[7]
                black_result.set(str(out[0])+' %  @  '+out[4])
                grey_result.set(str(out[1])+' %  @  '+out[5])
                total_result.set(str(out[2])+' %')
                save_ext.set('Mesh_'+cut_mode.get()+'_cut('+str(num_kolon)+'x'+str(num_vrstic)+')_')
                if final_var.get()=='final_ON':
                    show_analyzed()
                if autosave_var.get()== 'autosave_on':
                    save(mapa, fajl, save_ext.get())
                return out   
            elif choice_mode.get()=='globalno':
                boom = Cut_histogram(mapa, fajl, X_sections, Y_sections, color_to_count)
                save_ext.set(cut_mode.get()+'_global_')
                out = change(mapa, fajl, boom[0],boom[1])          # Pri GLOBALNEM načinu ne dobiš minimalnega THRESHA !
                if final_var.get()=='final_ON':
                    show_analyzed()
                if autosave_var.get()== 'autosave_on':
                    save(mapa, fajl, save_ext.get())
                return out
    else:
        out = change(mapa,fajl, bltr.get(), grtr.get())
        save_ext.set('Manual('+str(bltr.get())+','+str(grtr.get())+')_')
        if final_var.get()=='final_ON':
            show_analyzed()
        if autosave_var.get()== 'autosave_on':
            save(mapa, fajl, save_ext.get())
        return out

    
 
#   -------------        --------           -----------       -----     ---   ---   ---   ---  -------

# SAVE lines -------     ---------       ------        -------         ----------

out=None; out_opencv=None
def save(mapa, fajl, extension):
    izstopna_končnica = ".png"
    if bool(dataimage_var.get())is False:
        new_file = extension + fajl
    else:
        new_file = 'user_'+dataimage_var.get()+'_'+fajl

    if ".TIF" in fajl:
        new_file = new_file.replace(".TIF", izstopna_končnica)
    elif ".JPG" in fajl:
        new_file = new_file.replace(".JPG", izstopna_končnica)
  
    if shrani_kot.get()=='comma separated file':
        csv_file = mapa+'/'+datafile_var.get()+'.csv'
        with open(csv_file, mode=save_mode.get())as rezultat:
            zapisovalec = csv.writer(rezultat, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        emptiness=False
        with open(csv_file, 'r') as csvfile:
            csv_dict = [row for row in csv.DictReader(csvfile)]
            if len(csv_dict) == 0:
                emptiness=True

        if emptiness:
            with open(csv_file, mode=save_mode.get(), newline='')as rezultat:
                zapisovalec = csv.writer(rezultat, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                zapisovalec.writerow(['','SLIKA','','','','','metoda','način','','temna(%)','svetla(%)','skupno (%)','','thresh1','tresh2','','INTER','K-val','','columns','rows','','x1, x2','y1, y2','','Contrast','Brightness','Gamma'])                                                                                    
                zapisovalec.writerow([500*'-'])
                    
        with open(csv_file, mode= save_mode.get(), newline='')as rezultat:
            zapisovalec = csv.writer(rezultat, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if nacin.get()== 'auto' and cut_mode.get()=='opencv':
                if shranjuj_slike_var.get():
                    misc.imsave(mapa+'/'+new_file, out_opencv[3])
                zapisovalec.writerow([fajl, '','','','','',cut_mode.get(),'-','','-','-', out_opencv[2],'','-','-','','-',open_cv_var1.get(),'','-','-','',str(padX1.get())+', '+str(padX2.get()), str(padY1.get())+', '+str(padY2.get()),'', out_opencv[6][0], out_opencv[6][1], out_opencv[6][2]])                                                                              
            elif nacin.get()== 'auto' and cut_mode.get()!='opencv' and choice_mode.get()=='lokalno' and mesh_var.get()=='mesh_OFF':
                if shranjuj_slike_var.get():
                    misc.imsave(mapa+'/'+new_file, out[3])
                zapisovalec.writerow([fajl, '','','','','',cut_mode.get(),choice_mode.get(),'', out[0], out[1], out[2],'', out[4], out[5],'',inter_var.get(),'-','',str(num_kolon),str(num_vrstic),'',str(padX1.get())+', '+str(padX2.get()), str(padY1.get())+', '+str(padY2.get()),'', out[6][0], out[6][1], out[6][2]])
            elif nacin.get()== 'auto' and cut_mode.get()!='opencv' and choice_mode.get()=='lokalno' and mesh_var.get()=='mesh_ON':
                if shranjuj_slike_var.get():
                    misc.imsave(mapa+'/'+new_file, out[3])
                zapisovalec.writerow([fajl, '','','','','',cut_mode.get(),'zlepljeno','', out[0], out[1], out[2],'', '-', '-','',inter_var.get(),'-','',str(num_kolon),str(num_vrstic),'',str(padX1.get())+', '+str(padX2.get()), str(padY1.get())+', '+str(padY2.get()),'', out[6][0], out[6][1], out[6][2]])
            elif nacin.get()== 'auto' and cut_mode.get()!='opencv' and choice_mode.get()=='globalno':
                if shranjuj_slike_var.get():
                    misc.imsave(mapa+'/'+new_file, out[3])
                zapisovalec.writerow([fajl, '','','','','',cut_mode.get(),choice_mode.get(),'', out[0], out[1], out[2],'', out[4], out[5],'',inter_var.get(),'-','','-','-','',str(padX1.get())+', '+str(padX2.get()), str(padY1.get())+', '+str(padY2.get()),'', out[6][0], out[6][1], out[6][2]])
            elif nacin.get()== 'rocno':
                if shranjuj_slike_var.get():
                    misc.imsave(mapa+'/'+new_file, out[3])
                zapisovalec.writerow([fajl, '','','','','',nacin.get(),'-','', out[0], out[1], out[2],'', out[4], out[5],'','-','-','','-','-','',str(padX1.get())+', '+str(padX2.get()), str(padY1.get())+', '+str(padY2.get()),'', out[6][0], out[6][1], out[6][2]])

            
    elif shrani_kot.get() == 'text file':
        txt_file = mapa+'/'+datafile_var.get()+'.txt'
        with open(txt_file, save_mode.get())as txt_result:
            txt_result.write('')
        txt_prazno=False
        with open(txt_file, 'r')as beri:
            text_dict = [row for row in beri.read()]
            if len(text_dict) == 0:
                txt_prazno=True
        if txt_prazno:
            with open(txt_file, save_mode.get())as txt_result:
                txt_result.write('          SLIKA,     metoda,     način,     temna(%),     svetla(%),     skupno (%),     thresh1,     tresh2,     INTER,     K-val,     columns,     rows,     x1,     x2,     y1,     y2,     Contrast,     Brightness,     Gamma\n')
                txt_result.write(200*'-')
                txt_result.write('\r\n')
        with open(txt_file, save_mode.get())as txt_result:
            l=',     '; em='-     '
            if nacin.get()== 'auto' and cut_mode.get()=='opencv':
                if shranjuj_slike_var.get():
                    misc.imsave(mapa+'/'+new_file, out_opencv[3])
                txt_result.write(fajl+l+cut_mode.get()+l+em+l+em+l+em+l+str(out_opencv[2])+l+em+l+em+l+em+l+str(open_cv_var1.get())+l+em+l+em+l+str(padX1.get())+l+str(padX2.get())+l+str(padY1.get())+l+str(padY2.get())+l+out_opencv[6][0]+l+out_opencv[6][1]+l+out_opencv[6][2]+'\n')
            elif nacin.get()== 'auto' and cut_mode.get()!='opencv' and choice_mode.get()=='lokalno' and mesh_var.get()=='mesh_OFF':
                if shranjuj_slike_var.get():
                    misc.imsave(mapa+'/'+new_file, out[3])
                txt_result.write(fajl+l+cut_mode.get()+l+choice_mode.get()+l+str(out[0])+l+ str(out[1])+l+ str(out[2])+l+str(out[4])+l+str(out[5])+l+str(inter_var.get())+l+em+l+str(num_kolon)+l+str(num_vrstic)+l+str(padX1.get())+l+str(padX2.get())+l+ str(padY1.get())+l+str(padY2.get())+l+out[6][0]+l+out[6][1]+l+out[6][2]+'\n')
            elif nacin.get()== 'auto' and cut_mode.get()!='opencv' and choice_mode.get()=='lokalno' and mesh_var.get()=='mesh_ON':
                if shranjuj_slike_var.get():
                    misc.imsave(mapa+'/'+new_file, out[3])
                txt_result.write(fajl+l+cut_mode.get()+l+'zlepljeno'+l+str(out[0])+l+ str(out[1])+l+ str(out[2])+l+em+l+em+l+str(inter_var.get())+l+em+l+str(num_kolon)+l+str(num_vrstic)+l+str(padX1.get())+l+str(padX2.get())+l+ str(padY1.get())+l+str(padY2.get())+l+out[6][0]+l+out[6][1]+l+out[6][2]+'\n')
            elif nacin.get()== 'auto' and cut_mode.get()!='opencv' and choice_mode.get()=='globalno':
                if shranjuj_slike_var.get():
                    misc.imsave(mapa+'/'+new_file, out[3])
                txt_result.write(fajl+l+cut_mode.get()+l+choice_mode.get()+l+str(out[0])+l+str(out[1])+l+str(out[2])+l+str(out[4])+l+str(out[5])+l+str(inter_var.get())+l+em+l+em+l+em+l+str(padX1.get())+l+str(padX2.get())+l+str(padY1.get())+l+str(padY2.get())+l+out[6][0]+l+out[6][1]+l+out[6][2]+'\n')
            elif nacin.get()== 'rocno':
                if shranjuj_slike_var.get():
                    misc.imsave(mapa+'/'+new_file, out[3])
                txt_result.write(fajl+l+nacin.get()+l+em+l+str(out[0])+l+str(out[1])+l+str(out[2])+l+str(out[4])+l+str(out[5])+l+em+l+em+l+em+l+em+l+str(padX1.get())+l+str(padX2.get())+l+str(padY1.get())+l+str(padY2.get())+l+out[6][0]+l+out[6][1]+l+out[6][2]+'\n')

def destroy(x):
    if x:
        x.destroy()
        del x

tao=None; bck_save='#ffa366'; save_as_counter=0
def Shrani_kot():
    global tao
    destroy(tao)
    tao = Toplevel(root)
    tao.resizable(0,0)
    tao.title('Nastavitve shranjevanja')
    tao.config(bg=bck_save)
    Label(tao, bg=bck_save, height=3, width=10).grid(row=0, column=0)
    Label(tao, bg=bck_save, height=3, width=7).grid(row=0, column=2)
    Label(tao, text='Shranjuj podatke kot:', font=('Arial', '14'), bg=bck_save, fg='#6e6eaa'). grid(row=1, column=1)
    Label(tao, bg=bck_save, height=1, width=10).grid(row=2, column=0)
    Label(tao, text='Izberi imena datotek:', font=('Arial', '14'), bg=bck_save, fg='#6e6eaa'). grid(row=2, column=3)
    Label(tao, bg=bck_save, height=3, width=10).grid(row=0, column=5)
    Radiobutton(tao, text='Comma Separated File (.CSV)', command= lambda: datafile_ext.set('.csv'), indicatoron=0, selectcolor='#00cc99', variable=shrani_kot, value='comma separated file', fg='black', bg='#cc6699', font=('Arial', '20'), width=25, height=1).grid(row=3, column=1, columnspan=1, sticky=W) 
    Label(tao, bg=bck_save, height=1, width=10).grid(row=4, column=0)
    Radiobutton(tao, text='Text File (.TXT)', command= lambda: datafile_ext.set('.txt'), indicatoron=0, selectcolor='#00cc99', variable=shrani_kot, value='text file', fg='black', bg='#cc6699', font=('Arial', '20'), width=25, height=1).grid(row=5, column=1, columnspan=1, sticky=W) 
    ent11 = Entry(tao, textvariable=datafile_var, font=('Arial', '26'), width=22, justify='right', fg='blue'); ent11.grid(row=3, column=3, rowspan=3, sticky=W)
    Label(tao, textvariable=datafile_ext,bg=bck_save, fg='#6e6eaa', font=('Arial', '26')).grid(row=3, column=4, rowspan=3, sticky=W)
    Label(tao, bg=bck_save, height=2, width=10).grid(row=6, column=0)
    Label(tao, text='Naj poleg podatkov shranjujem tudi slike?', font=('Arial', '14'), bg=bck_save, fg='#6e6eaa', height=1).grid(row=7, column=1)
    Checkbutton(tao, text='Shranjuj slike', font=('Arial', '20'), variable = shranjuj_slike_var, onvalue = 1, offvalue = 0, indicatoron=0, selectcolor='#00cc99', bg='#cc6699', width=25, height=1).grid(row=8, column=1, columnspan=1, sticky=W)
    Label(tao, text='user_', bg=bck_save, fg='#6e6eaa', font=('Arial', '26'), width=5, anchor='w').grid(row=8, column=3, columnspan=1, sticky=W, padx=5)
    ent22 = Entry(tao, textvariable=dataimage_var, font=('Arial', '26'), width=17, justify='right', fg='blue'); ent22.grid(row=8, column=3, sticky=E)
    Label(tao, text='_(original filename).png', bg=bck_save, fg='#6e6eaa', font=('Arial', '26')).grid(row=8, column=4, sticky=W)
    Label(tao, bg=bck_save, height=5, width=10).grid(row=9, column=0)
    Button(tao, text='V redu', font=('Arial','25'), bg='#6e6eaa', fg='white', command=lambda: destroy(tao), width=25, height=1).grid(row=10, column=2, columnspan=2, sticky=W)
    Label(tao, bg=bck_save, height=3, width=10).grid(row=11, column=0)


# BRISANJE  DATOTEK -------------------------------------------------------------------

def pick_rootdir_DEL():
    global mape_del, folders_del
    master.directory = askdirectory()
    deldir_var.set(master.directory)
    mape_del=[]; folders_del = []; datoteke_del = []
    for subdir, dirs, files in os.walk(deldir_var.get()):
        mape_del.append(subdir)
        folders_del.append(dirs)
        datoteke_del.append(files)
    show_subfolders_DEL(folders_del[0])

amg_del=None
def show_subfolders_DEL(subfold):
    global amg_del, var_list_del, ER_count_var
    f ={}
    var_list_del = [IntVar(value=1) for i in range(len(subfold))]
    slejvi = [i for i in reversed(delfold.grid_slaves())]
    Label(delfold, bg=bck_del, height=3, width=10).grid(row=0, column=0)
    zamik=1
    for i in range(len(slejvi)):
        if i > zamik:
            slejvi[i].grid_remove()
    for x in range(len(subfold)):
        f["chkb{}".format(x)]=Checkbutton(delfold, text=subfold[x], variable = var_list_del[x], onvalue = 1, offvalue = 0, indicatoron=0, selectcolor='#00cc99', bg='#cc6699', width=31, font=('Arial','12'))
        amg_del=Namespace(**f)
        zamik2=10
        if x<zamik2:    
            exec("amg_del.chkb{0}.grid(row=x+zamik2, column=0, sticky=W)".format(x))
        else:
            exec("amg_del.chkb{0}.grid(row=x, column=1, sticky=E)".format(x))
    Label(delfold, bg=bck_del, height=2).grid(row=x+zamik2+1, column=0)
    Button(delfold, text='Odznači Vse', font=('Arial','12'), bg='#8A8A8A', fg='white', command= Untick_Delete, width=15).grid(row=x+zamik2+2, column=0, columnspan=1, sticky=E)
    Button(delfold, text='Izberi Vse', font=('Arial','12'), bg='#8A8A8A', fg='white', command= Mark_Delete, width=15).grid(row=x+zamik2+2, column=0, columnspan=1, sticky=W)
    Label(delfold, bg=bck_del, height=2).grid(row=x+zamik2+3, column=0)
    Label(delfold, bg=bck_del, height=3, width=8).grid(row=0, column=3)
    ER_count_var = StringVar()
    Label(delfold, textvariable=ER_count_var, bg=bck_del, height=3, font=('Arial', '22', 'bold'), fg='#000099').grid(row=100, column=0, rowspan=2, columnspan=2)

def Untick_Delete():
    for i in var_list_del:
            i.set(0)

def Mark_Delete():
    for i in var_list_del:
            i.set(1)

def Untick_All_Del():
        for i in delete_list:
            i.set(0)

def Mark_All_Del():
    for i in delete_list:
        i.set(1)

def Del_All_Entries():
    for i in entry_list:
        i.set('')

bck_del = '#66ccff'
btn_fg='black'

master=None
def Brisi():
    global master, art, delete_list, entry_list, delfold
    if master:
        master.destroy()
        del master
    master = Toplevel(root)
    master.resizable(0,0)
    master.title('Brisanje datotek')
    master.config(bg=bck_del)
    brisanje = Frame(master, bg=bck_del)
    brisanje.grid(row=0, column=0, sticky=NW)
    delfold = Frame(master, bg=bck_del)
    delfold.grid(row=0, column=1, sticky=NW)
    del_font = ('Arial', '24')

    opcije = ['.txt', '.csv', 'Manual', 'thresh_global', 'otsu_global', 'OpenCV', 'thresh_cut', 'otsu_cut', 'user_', 'Thresh_'] 
    g ={}
    delete_list = [IntVar(value=0) for i in range(len(opcije))]
    entry_list = [StringVar()for i in range(len(opcije))]

    Label(brisanje, bg=bck_del, height=3, width=10).grid(row=0, column=0)
    Label(brisanje, text='Osnovne predpone', font=('Arial', '14'), bg=bck_del). grid(row=1, column=1)
    Label(brisanje, text='Dodatne predpone', font=('Arial', '14'), bg=bck_del). grid(row=1, column=2, sticky=W)
    Label(brisanje, bg=bck_del, height=3, width=10).grid(row=0, column=3)

    for x in range(len(opcije)):
            g["vnos{}".format(x)]=Entry(brisanje, textvariable=entry_list[x], font=('Arial', '31'), width=20, justify='center', fg='blue')
            g["cbop{}".format(x)]=Checkbutton(brisanje, text=opcije[x], font=del_font, variable = delete_list[x], onvalue = 1, offvalue = 0, indicatoron=0, selectcolor='#00cc99', bg='#cc6699', width=15)
            art=Namespace(**g)
            exec("art.vnos{0}.grid(row=x+2,column=2, sticky=W)".format(x))
            exec("art.cbop{0}.grid(row=x+2, column=1, sticky=W)".format(x))

    Label(brisanje, bg=bck_del, height=2).grid(row=x+3, column=0)
    Button(brisanje, text='Odznači Vse', font=('Arial','12'), bg='#8A8A8A', fg='white', command=Untick_All_Del, width=15).grid(row=x+4, column=1, columnspan=1, sticky=E)
    Button(brisanje, text='Izberi Vse', font=('Arial','12'), bg='#8A8A8A', fg='white', command=Mark_All_Del, width=15).grid(row=x+4, column=1, columnspan=1, sticky=W)
    Button(brisanje, text='Briši dodatne predpone', font=('Arial','12'), bg='#8A8A8A', fg='white', command=Del_All_Entries, width=25).grid(row=1, column=2, sticky=E)
    Label(brisanje, bg=bck_del, height=1).grid(row=x+5, column=0)

    def Eraser():
        global art, erase_this, subs_to_delete
        erase_this = []
        for i in range(len(opcije)):
            wid_chk=str(eval("art.cbop{0}.cget('variable')".format(i)))
            chkb_val = eval("art.cbop{0}.getvar(wid_chk)".format(i))
            if chkb_val == 1:
                wid_ent=eval("art.vnos{0}.cget('text')".format(i))
                osnovna_predpona=eval("art.cbop{0}.cget('text')".format(i))
                dodatna_predpona=eval("art.vnos{0}.getvar(wid_ent)".format(i))
                erase_this.append((osnovna_predpona, dodatna_predpona))

        subs_to_delete = []

        switch='Go!'
        if not amg_del:
            pick_rootdir_DEL()
            switch='STOP'

        if switch=='Go!':
            for i in range(len(folders_del[0])):
                wid_but=eval("amg_del.chkb{0}.cget('text')".format(i))
                wid_chk=str(eval("amg_del.chkb{0}.cget('variable')".format(i)))
                chkb_val = eval("amg_del.chkb{0}.getvar(wid_chk)".format(i))
                if chkb_val == 1:
                    subs_to_delete.append(wid_but)

            rootdir = deldir_var.get(); erase_counter=0
            for pod in subs_to_delete:
                wf = rootdir+'/'+pod+'/Brez merila'
                print('    ..deleting files in folder:   ',wf); print(); print('*****************************')
                for filename in os.listdir(wf):
                    for hh in erase_this:
                        if all(kk in filename for kk in hh):
                            os.remove(wf+'/'+filename)
                            erase_counter+=1
        ER_count_var.set('Izbrisano: '+str(erase_counter)+'  datotek!')

    Delete_button = Button(brisanje, text='Briši', command=Eraser, font=('Arial', '19', 'bold'), width=8, height=2); Delete_button.grid(row=x+4, column=2, columnspan=1, sticky=E)
    PickDelFol = Button(brisanje, text='Izberi Mapo', command=pick_rootdir_DEL, font=('Arial', '19', 'bold'), width=10, height=2); PickDelFol.grid(row=x+4, column=2, columnspan=1)
# -----------------------------------------------------------------------------------------------


""" GUI - Uporabniški vmesnik -----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

from tkinter import *
from tkinter.messagebox import *
from tkinter.filedialog import *

root=Tk()
root.geometry('1631x822+10+10')
root.resizable(0, 0) 
root.title('MultiPhaser 1.0 by Andro'+u'\u00AE')
bck='#666699'
root.config(bg=bck)
btn_font = ('Arial', '16')
btn_fg = 'black'

frstc='#5c5c8a'
frame1 = Frame(root, bg=bck)
frame1.grid(row=0, column=0, columnspan=2, sticky=NW)
frame1.config(borderwidth=3, relief=SUNKEN)

single_file, single_folder = None, None
SF=[None]; sf=[None]; album=[]
def izberi_sliko():
    global SF, sf, album
    if SF:
        root.filename =  askopenfilename(initialdir = SF[-1], title = "Izberi sliko", filetypes = (("TIF","*.tif"), ("JPEG","*.jpg"),("PNG","*.png")))#,("all files","*.*")))
    else:
        root.filename =  askopenfilename(initialdir = "C:/Users/akocjan/Desktop/MARTIN/testna mapa", title = "Izberi sliko", filetypes = (("TIF","*.tif"), ("JPEG","*.jpg"),("PNG","*.png")))#,("all files","*.*")))
    a=root.filename.split('/')
    single_file=a[-1]
    single_folder = "/".join(a[:-1])
    display1.config(fg='yellow')
    img = cv2.imread(root.filename,0)

    
    if raw_var.get()=='raw_ON':
        im= np.stack((img,img,img), axis=2)
        plt.imshow(im)
    if single_folder:
        SF.append(single_folder)
        sf.append(single_file)
        ime_slike.set(sf[-1])

    album.append(img)
    
    return SF[-1], sf[-1]

def prikazi_sliko(mapa, fajl):
    try:
        file = mapa+'/'+fajl
        img = cv2.imread(file,0)
        plt.figure(); plt.title('ORIGINAL'); plt.xlabel(fajl)
        im= np.stack((img,img,img), axis=2)
        plt.imshow(im)
    except TypeError:
        display1.config(fg='red')
        ime_slike.set('..prosim izberi sliko!')
        pass

# Contrast & Brightness Control ===> Histogram Manipulation!

def CB(mapa,fajl, alpha, beta, gamma):
    try:
        file = mapa+'/'+fajl
        img = cv2.imread(file,0)
        ni = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        gam_cor = Gamma_Correction(ni, gamma=gamma)
        img_CB = np.stack((gam_cor, gam_cor, gam_cor), axis=2)
        plt.figure(); plt.title('Contrast: '+str(alpha)+'   Brightness: '+str(beta)+'   Gamma: '+str(gamma))
        plt.xlabel(fajl)
        plt.imshow(img_CB)
    except TypeError:
        display1.config(fg='red')
        ime_slike.set('..prosim izberi sliko!')
        pass

# Some Tkinter Variables..
#---------------------------------------------------------------------
SM=StringVar(); SM.set('closed')
save_ext = StringVar(); save_ext.set('default_')
ime_slike = StringVar()

datafile_ext = StringVar(); datafile_ext.set('.csv')
shrani_kot = StringVar(); shrani_kot.set('comma separated file')
datafile_var=StringVar(); datafile_var.set('Multi Fazna Analiza')
shranjuj_slike_var=IntVar(value=1)
dataimage_var=StringVar(); dataimage_var.set('')

#---------------------------------------------------------------------

l1 = Label(frame1, bg=bck, text='Program za analizo deleža sekundarnih faz iz metalografskih posnetkov', fg='white'); l1.grid(row=0, column=0, columnspan=4)

b0 = Button(frame1, text='IZBERI SLIKO', command=izberi_sliko, fg=btn_fg, font=('Arial','18'), width=24); b0.grid(row=1, column=0, columnspan=4, sticky=W)
b1 = Button(frame1, text='ORIGINAL', command=lambda: prikazi_sliko(SF[-1], sf[-1]), fg=btn_fg, font=('Arial','17'), width=17); b1.grid(row=1, column=1, columnspan=4)
b2 = Button(frame1, text='KONČNA', command=show_analyzed, fg=btn_fg, font=('Arial','17', 'bold'), width=16); b2.grid(row=1, column=2, columnspan=4, sticky=E)

display1 = Label(frame1, textvariable=ime_slike, bg='black', fg='yellow', font=('Arial', '25'), width=42, height=2, borderwidth=6, relief="solid"); display1.grid(row=2, column=0, columnspan=4)

black_result = StringVar()
display2 = Label(frame1, textvariable=black_result, bg='black', fg='#00B3FF', width=16, height=2, font=('Arial','23', 'bold')); display2.grid(row=3, column=1, sticky=W)
display3 = Label(frame1, text='DARK:  ', bg='black', fg='#00B3FF', width=12, height=4, font=('Arial','12')); display3.grid(row=3, column=0, sticky=W)

grey_result = StringVar()
display4 = Label(frame1, textvariable=grey_result, bg='black', fg='#00B3FF', width=16, height=2, font=('Arial','23', 'bold')); display4.grid(row=4, column=1, sticky=W)
display5 = Label(frame1, text='BRIGHT:  ', bg='black', fg='#00B3FF', width=12, height=4, font=('Arial','12')); display5.grid(row=4, column=0, sticky=W)

total_result = StringVar()
display6 = Label(frame1, textvariable=total_result, bg='black', fg='lime', width=9, height=3, font=('Arial','32', 'bold'), borderwidth=0, relief="solid"); display6.grid(row=3, column=3, sticky=W, rowspan=2)
display7 = Label(frame1, text='  TOTAL:', bg='black', fg='lime', width=9, height=5, font=('Arial','19'), borderwidth=7, relief="solid"); display7.grid(row=3, column=2, sticky=W, rowspan=2)

frame2 = Frame(root, bg=bck)
frame2.grid(row=1, column=0, sticky=NW)
frame2.config(borderwidth=4, relief=SUNKEN)

nacin = StringVar(); nacin.set('auto')
rdbt1=Radiobutton(frame2, text='MANUAL & Correction', command=lambda: nacin.set('rocno'), indicatoron=0, selectcolor='#00cc99', variable=nacin, value='rocno', fg=btn_fg, bg='#cc6699', font=btn_font, width=33); rdbt1.grid(row=0, column=0, columnspan=4, sticky=W) 

Label(frame2, height=1, bg=bck).grid(row=1, column=0)
Label(frame2, text='Dark threshold:', fg='white', font=('Arial', '12'), bg=bck   , width=15).grid(row=2, column=1, columnspan=3, sticky=W, padx=60)
Label(frame2, bg=bck).grid(row=1, column=3)
bltr = IntVar()
e1=Entry(frame2, textvariable=bltr, font=('Arial', '36'), width=5, bg='white', fg='blue', justify='center'); e1.grid(row=3 , column=1, columnspan=3, sticky=W, padx=60)

Label(frame2, height=1, bg=bck).grid(row=4, column=0)
Label(frame2, text='Bright threshold:', fg='white', font=('Arial', '12'), bg=bck   , width=15).grid(row=5, column=1, columnspan=3, sticky=W, padx=60)
grtr = IntVar()
e2=Entry(frame2, textvariable=grtr, font=('Arial', '36'), width=5, bg='white', fg='blue', justify='center'); e2.grid(row=7 , column=1, columnspan=3, sticky=W, padx=60)
Label(frame2, height=3, bg=bck).grid(row=8, column=0)

contrast_var=IntVar(); contrast_var.set(-100)
Label(frame2, text='Contrast', bg=bck, fg='white', font=('Arial', '12')).grid(row=9, column=0)
e93=Entry(frame2, textvariable=contrast_var, font=('Arial', '25'), width=5, bg='white', fg='blue', justify='center'); e93.grid(row=10 , column=0)

brightness_var=DoubleVar(); brightness_var.set(1.5)
Label(frame2, text='Brightness', bg=bck, fg='white', font=('Arial', '12')).grid(row=9, column=1)
e87=Entry(frame2, textvariable=brightness_var, font=('Arial', '25'), width=5, bg='white', fg='blue', justify='center'); e87.grid(row=10 , column=1)

gamma_var=DoubleVar(); gamma_var.set(1.0)
Label(frame2, text='Gamma', bg=bck, fg='white', font=('Arial', '12')).grid(row=9, column=2)
e516=Entry(frame2, textvariable=gamma_var, font=('Arial', '25'), width=5, bg='white', fg='blue', justify='center'); e516.grid(row=10 , column=2, sticky=W)

CB_var=IntVar(); CB_var.set(0)
CB_it = Checkbutton(frame2, fg='#ffcc99',selectcolor=bck, text = "Use Corr.", variable = CB_var, onvalue = 1, offvalue = 0, indicatoron=1, font=('Arial','14'), bg=bck); CB_it.grid(row=9, column=3)

b99 = Button(frame2, text='Corr. Image', command=lambda:CB(SF[-1], sf[-1], brightness_var.get(), contrast_var.get(), gamma_var.get()), fg=btn_fg, font=('Arial','16'), width=9); b99.grid(row=10, column=3, sticky=E)


frame3 = Frame(root, bg=bck)
frame3.grid(row=1, column=1, sticky=NE)
frame3.config(borderwidth=4, relief=SUNKEN)

rdbt2=Radiobutton(frame3, text='AUTO', command=lambda: nacin.set('auto'), indicatoron=0, selectcolor='#00cc99', variable=nacin, value='auto', fg=btn_fg, bg='#cc6699', font=btn_font, width=32); rdbt2.grid(row=0, column=0, columnspan=3, sticky=W) 
Label(frame3, height=1, bg=bck).grid(row=1, column=2)

choice_mode=StringVar(); choice_mode.set('lokalno')
glob_but=Radiobutton(frame3, text='Global', command=lambda: choice_mode.set('globalno'), variable=choice_mode, value='globalno', bg=bck, font=('Arial','18')); glob_but.grid(row=2, column=0)
cut_but=Radiobutton(frame3, text='Cut   ', command=lambda: choice_mode.set('lokalno'), variable=choice_mode, value='lokalno', bg=bck, font=('Arial','18')); cut_but.grid(row=2, column=1)
mesh_var=StringVar(); mesh_var.set('mesh_OFF')
do_mesh = Checkbutton(frame3, text = "Mesh", variable = mesh_var, onvalue = 'mesh_ON', offvalue = 'mesh_OFF', bg=bck, indicatoron=1, font=('Arial','18')); do_mesh.grid(row=2, column=2)

global_minimum = IntVar(); global_minimum.set(130)
Label(frame3, text='Global Min:', fg='white', font=('Arial', '12'), bg=bck).grid(row=3, column=0)
e111 = Entry(frame3, textvariable=global_minimum, font=('Arial', '12'), width=6, bg='white', fg='blue', justify='center'); e111.grid(row=4 , column=0)

inter_var = DoubleVar(); inter_var.set(0.0)
Label(frame3, text='Inter Value:', fg='white', font=('Arial', '12'), bg=bck).grid(row=5, column=0)
e112 = Entry(frame3, textvariable=inter_var, font=('Arial', '12'), width=6, bg='white', fg='blue', justify='center'); e112.grid(row=6 , column=0)

open_cv_var1 = IntVar(); open_cv_var1.set(10)
Label(frame3, text='OpenCV K-val:', fg='white', font=('Arial', '12'), bg=bck).grid(row=8, column=0)
e4 = Entry(frame3, textvariable=open_cv_var1, font=('Arial', '12'), width=6, bg='white', fg='blue', justify='center'); e4.grid(row=9 , column=0)

num_Xsections = IntVar(); num_Xsections.set(20)
Label(frame3, text='X-cuts:', fg='white', font=('Arial', '12'), bg=bck).grid(row=3, column=1)
e3=Entry(frame3, textvariable=num_Xsections, font=('Arial', '25'), width=6, bg='white', fg='blue', justify='center'); e3.grid(row=4 , column=1, rowspan=2)

num_Ysections = IntVar(); num_Ysections.set(20)
Label(frame3, text='Y-cuts:', fg='white', font=('Arial', '12'), bg=bck).grid(row=7, column=1)
e5=Entry(frame3, textvariable=num_Ysections, font=('Arial', '25'), width=6, bg='white', fg='blue', justify='center'); e5.grid(row=8 , column=1, rowspan=2)

padX1 = DoubleVar(); padX1.set(0.0)
e6=Entry(frame3, textvariable=padX1, font=('Arial', '12'), width=6, bg='white', fg='blue', justify='center'); e6.grid(row=4 , column=2, sticky=N)
Label(frame3, text='x1 %', fg='white', font=('Arial', '12'), bg=bck).grid(row=4, column=2, sticky=NE)

padX2 = DoubleVar(); padX2.set(100.0)
e7=Entry(frame3, textvariable=padX2, font=('Arial', '12'), width=6, bg='white', fg='blue', justify='center'); e7.grid(row=5 , column=2, sticky=S)
Label(frame3, text='x2 %', fg='white', font=('Arial', '12'), bg=bck).grid(row=5, column=2, sticky=SE)

padY1 = DoubleVar(); padY1.set(0.0)
e8=Entry(frame3, textvariable=padY1, font=('Arial', '12'), width=6, bg='white', fg='blue', justify='center'); e8.grid(row=8 , column=2, sticky=N)
Label(frame3, text='y1 %', fg='white', font=('Arial', '12'), bg=bck).grid(row=8, column=2, sticky=NE)

padY2 = DoubleVar(); padY2.set(100.0)
e9=Entry(frame3, textvariable=padY2, font=('Arial', '12'), width=6, bg='white', fg='blue', justify='center'); e9.grid(row=9 , column=2, sticky=S)
Label(frame3, text='y2 %', fg='white', font=('Arial', '12'), bg=bck).grid(row=9, column=2, sticky=SE)

Label(frame3, height=1, bg=bck).grid(row=10, column=1)

cut_mode=StringVar(); cut_mode.set('thresh')
thresh_but=Radiobutton(frame3, text='Thresh', command=lambda: cut_mode.set('thresh'), selectcolor='#00cc99', bg='#cc6699', indicatoron=0, variable=cut_mode, value='thresh', font=('Arial','19'), width=8); thresh_but.grid(row=11, column=0, sticky=W)
otsu_but=Radiobutton(frame3, text=' Otsu ', command=lambda: cut_mode.set('otsu'), selectcolor='#00cc99', bg='#cc6699', indicatoron=0, variable=cut_mode, value='otsu', font=('Arial','19'), width=8); otsu_but.grid(row=11, column=1, sticky=W)
opencv_but=Radiobutton(frame3, text='OpenCV', command=lambda: cut_mode.set('opencv'), selectcolor='#00cc99', bg='#cc6699', indicatoron=0, variable=cut_mode, value='opencv', font=('Arial','19'), width=8); opencv_but.grid(row=11, column=2,sticky=E)

cut_hist = Button(frame3, text='Histogram', command=lambda:Show_Histogram(SF[-1], sf[-1], num_Xsections.get(), num_Ysections.get(), purple), fg=btn_fg, font=('Arial', '16'), width=10); cut_hist.grid(row=12, column=0, sticky=W)
cut_vsTresh = Button(frame3, text='% vs. tresh', command=lambda:thresh_from_percent(SF[-1], sf[-1], num_Xsections.get(), num_Ysections.get(), purple), fg=btn_fg, font=('Arial', '16'), width=10); cut_vsTresh.grid(row=12, column=1, sticky=W)
cut_Mesh = Button(frame3, text='Show Mesh', command=lambda:Show_Mesh(SF[-1], sf[-1], num_Xsections.get(), num_Ysections.get(), purple), fg=btn_fg, font=('Arial', '16'), width=10); cut_Mesh.grid(row=12, column=2, sticky=E)

frame4 = Frame(root, bg=bck)
frame4.grid(row=2, column=0, sticky=NW, columnspan=2)

RUN_button = Button(frame4, text='RUN', command=lambda:BIG_RUN(SF[-1], sf[-1], num_Xsections.get(), num_Ysections.get(), purple), fg=btn_fg, font=('Arial', '30', 'bold'), width=17, height=1); RUN_button.grid(row=7, column=0, sticky=W, columnspan=4, rowspan=1)

raw_var=StringVar(); raw_var.set('raw_OFF')
show_raw = Checkbutton(frame4, text = "Show Raw", variable = raw_var, onvalue = 'raw_ON', offvalue = 'raw_OFF', selectcolor='#00cc99', bg='#cc6699', width = 9, indicatoron=0, font=('Arial','18')); show_raw.grid(row=6, column=0, sticky=NW, columnspan=1)

showhist_var=StringVar(); showhist_var.set('hist_OFF')
show_hist = Checkbutton(frame4, text = "Show Hist", variable = showhist_var, onvalue = 'hist_ON', offvalue = 'hist_OFF', selectcolor='#00cc99', bg='#cc6699', width = 9, indicatoron=0, font=('Arial','18')); show_hist.grid(row=6, column=1, sticky=N, columnspan=1)

final_var=StringVar(); final_var.set('final_ON')
show_final = Checkbutton(frame4, text = "Show Final", variable = final_var, onvalue = 'final_ON', offvalue = 'final_OFF', selectcolor='#00cc99', bg='#cc6699', width = 9, indicatoron=0, font=('Arial','18')); show_final.grid(row=6, column=2, sticky=E, columnspan=1)

Label(frame4, height=1, bg=bck, width=1).grid(row=6, column=3, sticky=W)


autosave_var = StringVar(); autosave_var.set('autosave_off')
chbut = Checkbutton(frame4, text = "AutoSave", variable = autosave_var, onvalue = 'autosave_on', offvalue = 'autosave_off', selectcolor='#00cc99', bg='#cc6699', width = 8, indicatoron=0, font=('Arial','19')); chbut.grid(row=6, column=4, sticky=NW, columnspan=2)

save_mode=StringVar(); save_mode.set('a')
add_but=Radiobutton(frame4, text='  Add   ', command=lambda: save_mode.set('a'), selectcolor='#6699ff', bg='#3333cc', indicatoron=0, variable=save_mode, value='a', font=('Arial','19'), width=8); add_but.grid(row=6, column=6, sticky=N, columnspan=2)
rewrite_but=Radiobutton(frame4, text='Rewrite', command=lambda: save_mode.set('w'), selectcolor='#6699ff', bg='#3333cc', indicatoron=0, variable=save_mode, value='w', font=('Arial','19'), width=8); rewrite_but.grid(row=6, column=8, sticky=NE, columnspan=2)

SAVE_button = Button(frame4, text='SAVE', command=lambda: save(SF[-1], sf[-1], save_ext.get()), fg=btn_fg, font=('Arial', '19', 'bold'), width=8, height=2); SAVE_button.grid(row=7, column=4, sticky=SW, columnspan=3)
SaveAs_button = Button(frame4, text='Save As', command=Shrani_kot, fg=btn_fg, font=('Arial', '19', 'bold'), width=8, height=2); SaveAs_button.grid(row=7, column=6, sticky=N, columnspan=2)

deldir_var=StringVar()

  
Delete_button = Button(frame4, text='Delete', command=Brisi, font=('Arial', '19', 'bold'), width=8, height=2); Delete_button.grid(row=7, column=8, sticky=NE, columnspan=2)

""" AUTO-CONTROL """

bck_auto = '#669999'
frame5 = Frame(root, bg=bck_auto)
frame5.grid(row=0, column=2, sticky="nsew", columnspan=2, rowspan=6)
frame5.config(borderwidth=4, relief=SUNKEN)
root.grid_rowconfigure(0, weight=1)

def pick_rootdir():
    global mape, folders
    root.directory = askdirectory()
    rootdir_var.set(root.directory)
    mape=[]; folders = []; datoteke = []
    for subdir, dirs, files in os.walk(rootdir_var.get()):
        mape.append(subdir)
        folders.append(dirs)
        datoteke.append(files)
    show_subfolders(folders[0])

amg=None
def show_subfolders(subfold):
    global amg, var_list
    d={}; stopnica = 13
    var_list = [IntVar(value=1) for i in range(len(subfold))]
    slejvi = [i for i in reversed(frame5.grid_slaves())]
    for i in range(len(slejvi)):
        if i > 12:
            slejvi[i].grid_remove()
    for x in range(len(subfold)):
        d["gumb{}".format(x)]=Button(frame5, text=subfold[x], command= lambda x=x: Auto_Run_Subfolder(subfold[x], num_Xsections.get(), num_Ysections.get(), purple, 'analyzed_', all_switch=False), font=('Arial','12'), width=36)
        d["chkb{}".format(x)]=Checkbutton(frame5, variable = var_list[x], onvalue = 1, offvalue = 0,  bg=bck_auto)
        amg=Namespace(**d)
        if x<stopnica:
            exec("amg.gumb{0}.grid(row=x+stopnica,column=0, columnspan=3)".format(x))
            exec("amg.chkb{0}.grid(row=x+stopnica, column=0, sticky=W)".format(x))
        else:
            exec("amg.gumb{0}.grid(row=x,column=3, columnspan=3)".format(x))
            exec("amg.chkb{0}.grid(row=x, column=3, sticky=W)".format(x))
    Label(frame5, bg=bck_auto, height=2).grid(row=x+stopnica+1, column=0)
    Button(frame5, text='Odznači Vse', font=('Arial','12'), bg='#8A8A8A', fg='white', command= Untick_All_Auto, width=12).grid(row=x+stopnica+2, column=2, columnspan=2, sticky=W)
    Button(frame5, text='Izberi Vse', font=('Arial','12'), bg='#8A8A8A', fg='white', command= Mark_All_Auto, width=12).grid(row=x+stopnica+2, column=3, columnspan=2, sticky=W)
    Label(frame5, bg=bck_auto, height=2).grid(row=x+stopnica+3, column=0)
    
    
def Untick_All_Auto():
        for i in var_list:
            i.set(0)

def Mark_All_Auto():
    for i in var_list:
            i.set(1)

Label(frame5, bg=bck_auto, text='Inštitut za materiale in tehnologije (IMT), Ljubljana', fg='white').grid(row=0, column=0, columnspan=6)

ba0 = Button(frame5, text='IZBERI MAPO', command=pick_rootdir, fg=btn_fg, font=('Arial','18'), width=28); ba0.grid(row=1, column=0,sticky=W, columnspan=4)
ba1 = Button(frame5, text='AUTO RUN', command=lambda: AUTO_RUN_ALL('analyzed_'), fg=btn_fg, font=('Arial','18'), width=28); ba1.grid(row=1, column=3, sticky=E, columnspan=4)

Label(frame5, bg='black', text='ROOTDIR:', fg='#00ccff', font=('Arial', '12'), width=15, height=2).grid(row=2, column=0, sticky=W)
rootdir_var = StringVar()
Label(frame5, bg='black', textvariable=rootdir_var, fg='#00ccff', font=('Arial', '12'), width=73, borderwidth=2, relief="solid", height=2, anchor="w").grid(row=2, column=1, columnspan=5, sticky=W)

Label(frame5, bg='black', text='SUBDIR:', fg='#9966ff', font=('Arial', '12'), width=15, height=2).grid(row=3, column=0, sticky=W)
subdir_var = StringVar()
Label(frame5, bg='black', textvariable=subdir_var, fg='#9966ff', font=('Arial', '12'), width=73, borderwidth=2, relief="solid", height=2, anchor="w").grid(row=3, column=1, columnspan=5, sticky=W)

Label(frame5, bg='black', text='FILE:', fg='#aaff00', font=('Arial', '12'), width=15, height=2).grid(row=4, column=0, sticky=W)
file_var = StringVar()
Label(frame5, bg='black', textvariable=file_var, fg='#aaff00', font=('Arial', '12'), width=73, borderwidth=2, relief="solid", height=2, anchor="w").grid(row=4, column=1, columnspan=5, sticky=W)

com1_var = StringVar()
Label(frame5, bg='black', textvariable=com1_var, fg='white', font=('Arial', '17', 'bold'), width=57, borderwidth=2, relief="solid", height=2).grid(row=5, column=0, columnspan=6, sticky=W)

com2_var = StringVar()
Label(frame5, bg='black', textvariable=com2_var, fg='white', font=('Arial', '17', 'bold'), width=57, height=2).grid(row=6, column=0, sticky=W, columnspan=6)

Label(frame5, bg=bck_auto).grid(row=7, column=0)

vstopna_končnica = ".tif"
izstopna_končnica = ".png"


def AUTO_RUN_ALL(extension):
    global subs_to_analyze, all_count, all_slik, all_zacetek
    subs_to_analyze = []
    if not amg:
        pick_rootdir()
    for i in range(len(folders[0])):
        wid_but=eval("amg.gumb{0}.cget('text')".format(i))
        wid_chk=str(eval("amg.chkb{0}.cget('variable')".format(i)))
        chkb_val = eval("amg.chkb{0}.getvar(wid_chk)".format(i))
        if chkb_val == 1:
            subs_to_analyze.append(wid_but)
        
    #print(subs_to_analyze)
    #print(len(subs_to_analyze))
    all_zacetek = time.time()

    if shrani_kot.get()=='comma separated file':
        with open(mape[0]+'/Povprečja.csv', mode=save_mode.get())as rezultat:
            zapisovalec = csv.writer(rezultat, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            zapisovalec.writerow(['MAPA','','','TEMNA(%)','','SVETLA(%)','','SKUPNO(%)'])

    elif shrani_kot.get() == 'text file':
        with open(mape[0]+'/Povprečja.txt', save_mode.get())as txt_result:
            txt_result.write(' \r\n')
            txt_result.write('     MAPA,     TEMNA(%),     SVETLA(%),     SKUPNO(%)\n')
            txt_result.write(75*'-'+'\n\r\n')
        

    all_slik = 0
    rootdir = mape[0]
    for pod in subs_to_analyze:
        wf = rootdir+'/'+pod+'/Brez merila'
        for filename in os.listdir(wf):
            if filename.endswith(vstopna_končnica )or filename.endswith(".JPG")and extension not in filename:
                all_slik+=1

    all_count=0
    for podmapa in subs_to_analyze:
        Auto_Run_Subfolder(podmapa, num_Xsections.get(), num_Ysections.get(), purple, extension, all_switch=True)


def Auto_Run_Subfolder(SUBDIR, X_sections, Y_sections, color_to_count, extension, all_switch):
    global all_count, all_slik, all_zacetek
    zacetek = time.time()
    counter=0; vseh_slik=0
    rootdir = mape[0]
    rts = rootdir+'/'+SUBDIR   
    for j in os.listdir(rts):
            if 'Brez merila' in j:
                    wf = rts+'/'+j
                    temna=[]; siva=[]; skupno =[]
                    for filename in os.listdir(wf):
                            if filename.endswith(vstopna_končnica )or filename.endswith(".JPG")and extension not in filename:
                                    vseh_slik+=1
                    com2_var.set('')
                    if all_switch and all_count==0:
                        com1_var.set('Obdelujem sliko '+str(1)+'/'+str(all_slik)+' ..')
                    elif all_switch is False and counter==0:
                        com1_var.set('Obdelujem sliko '+str(1)+'/'+str(vseh_slik)+' ..')
                    for filename in os.listdir(wf):
                            root.update()
                            subdir_var.set(SUBDIR+'     '+'('+str(vseh_slik)+' slik)')
                            if filename.endswith(vstopna_končnica )or filename.endswith(".JPG")and extension not in filename:    
                                    file_var.set(filename)
                                    counter+=1
                                    if all_switch:
                                        all_count+=1
                                        if all_count < all_slik:
                                            com1_var.set('Obdelujem sliko '+str(all_count+1)+'/'+str(all_slik)+' ..')
                                        else:
                                            com1_var.set('Obdelujem sliko '+str(all_count)+'/'+str(all_slik)+' ..')
                                    else:
                                        if counter < vseh_slik:
                                            com1_var.set('Obdelujem sliko '+str(counter+1)+'/'+str(vseh_slik)+' ..')
                                        else:
                                            com1_var.set('Obdelujem sliko '+str(counter)+'/'+str(vseh_slik)+' ..')

                                    start=time.time()
                                    ###############################################################
                                    gotovo = BIG_RUN(wf, filename, X_sections, Y_sections, color_to_count)
                                    ###############################################################
                                    black_result.set(str(gotovo[0])+' %  @  '+str(gotovo[4]))
                                    grey_result.set(str(gotovo[1])+' %  @  '+str(gotovo[5]))
                                    total_result.set(str(gotovo[2])+' %')
                                    display1.config(fg='yellow')
                                    ime_slike.set(filename)
                                    end=time.time()
                                    delta = end-start
                                    seconds = delta%60
                                    minutes = (delta % 3600)// 60
                                    if all_switch:
                                        y = (all_slik-all_count) * delta
                                    else:
                                        y = (vseh_slik-counter) * delta
                                    time_left=str(datetime.timedelta(seconds=y)).split(".")[0]
                                    konc_ura=time.strftime("%H:%M",time.localtime(end+y))
                                    min_left = int(time_left[2:].split(":")[0])
                                    sec_left = int(time_left[2:].split(":")[1])
                                    com2_var.set('Preostali čas:    '+str(min_left)+"'"+'  '+str(sec_left)+'"'+'    , analiza bo končana ob '+konc_ura)
                                    temna.append(gotovo[0])
                                    siva.append(gotovo[1])
                                    skupno.append(gotovo[2])
                    if cut_mode.get()!='opencv':
                        tem_avg = round(sum(temna)/len(temna), 1)
                        siv_avg = round(sum(siva)/len(siva), 1)
                    sku_avg = round(sum(skupno)/len(skupno), 1)

                    if shrani_kot.get()=='comma separated file':
                        csv_file = wf+'/'+datafile_var.get()+'.csv'
                        with open(csv_file, mode=save_mode.get())as rezultat:
                                zapisovalec = csv.writer(rezultat, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                if cut_mode.get()!='opencv':
                                    zapisovalec.writerow([''])
                                    zapisovalec.writerow(['','','','','','','','POVPREČJA','', tem_avg, siv_avg,sku_avg])
                                    zapisovalec.writerow([500*'-'])
                                else:
                                    zapisovalec.writerow([''])
                                    zapisovalec.writerow(['','','','','','','','POVPREČJA','','','',sku_avg])
                                    zapisovalec.writerow([500*'-'])
                        with open(rootdir+'/Povprečja.csv', mode=save_mode.get())as rezultat:
                                zapisovalec = csv.writer(rezultat, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                if cut_mode.get()!='opencv':
                                    zapisovalec.writerow([SUBDIR,'','',tem_avg,'',siv_avg,'',sku_avg])
                                else:
                                    zapisovalec.writerow([SUBDIR,'',sku_avg])

                    elif shrani_kot.get()=='text file':
                        txt_file = wf+'/'+datafile_var.get()+'.txt'
                        l=',     '; em='-     '
                        with open(txt_file, save_mode.get())as txt_result: 
                            if cut_mode.get()!='opencv':
                                txt_result.write('\n'); txt_result.write(20*' '+'POVPREČJA'+l+str(tem_avg)+l+str(siv_avg)+l+str(sku_avg)+'\n')
                                txt_result.write(200*'-'+'\n\r\n')
                            else:
                                txt_result.write('\n'); txt_result.write(20*' '+'POVPREČJA'+l+str(sku_avg)+'\n')
                                txt_result.write(200*'-'+'\n\r\n')
                        with open(rootdir+'/Povprečja.txt', save_mode.get())as txt_result:
                            if cut_mode.get()!='opencv':
                                txt_result.write(SUBDIR+l+str(tem_avg)+l+str(siv_avg)+l+str(sku_avg)+'\n')
                            else:
                                txt_result.write(SUBDIR+l+str(sku_avg)+'\n')
            
                      
    if all_switch and SUBDIR == subs_to_analyze[-1]:
        if shrani_kot.get() == 'text file':
            with open(rootdir+'/Povprečja.txt', save_mode.get())as txt_result:
                txt_result.write(75*'-'+'\n')
                txt_result.write(' \r\n')
        elif shrani_kot.get()=='comma separated file':
            with open(rootdir+'/Povprečja.csv', mode=save_mode.get())as rezultat:
                zapisovalec = csv.writer(rezultat, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                zapisovalec.writerow([120*'-'])
                zapisovalec.writerow([])
        com1_var.set('KONČANO!')
        konec = time.time()
        razlika = konec-all_zacetek
        sekund = int(razlika%60)
        minut = int((razlika % 3600)// 60)
        if minut==0:
            com2_var.set('Slik:   '+str(all_slik)+'       Čas:   '+str(sekund)+'"')
        elif minut >0:
            com2_var.set('Slik:   '+str(all_slik)+'       Čas:   '+str(minut)+"'  "+str(sekund)+'"')
        root.update()
        for i in range(3):
            beeper(1800, 1800, 100, 0.02, 100, 0.5)
    elif all_switch is False:
        com1_var.set('KONČANO!')
        konec = time.time()
        razlika = konec-zacetek
        sekund = int(razlika%60)
        minut = int((razlika % 3600)// 60)
        if minut==0:
            com2_var.set('Slik:   '+str(vseh_slik)+'      Čas:   '+str(sekund)+'"')
        elif minut >0:
            com2_var.set('Slik:   '+str(vseh_slik)+'      Čas:   '+str(minut)+"'  "+str(sekund)+'"')
        root.update()
        for i in range(3):
            beeper(1800, 1800, 100, 0.02, 100, 0.5)
        


# BEEPER
import winsound
def beeper(f1,f2,t1,t2,t3, t4):
    winsound.Beep(f1, t1)
    time.sleep(t2)
    winsound.Beep(f2, t3)
    time.sleep(t4)



# THE END


