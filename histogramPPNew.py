# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import cv2
# import cv
import matplotlib.pyplot as plt
from collections import deque
import itertools as it
import scipy.ndimage.measurements as sp
import skimage.measure as sk
import math
from PIL import Image
#import Image
import os
import plotly.plotly as py

def grid(im,height, width):
	def left():
		w=0
		b=0
		while w < width:
			h=0
			while h < height:
				if(im[h,w]!=255 and b!=255):
					#left=[h,w]
					left=w
					return left
					break
				b=im[h,w]
				h=h+1
			w=w+1

	def top():
		h=0
		b=0
		while h < height:
			w=0
			while w < width:
				if(im[h,w]!=255 and b!=255):
					top=h
					#top=[w,h]
					return top
					break
				b=im[h,w]
				w=w+1
			h=h+1

	def right():
		b=0
		w=width-1
		while w >= 0:
			h=height-1
			while h >= 0:
				if(im[h,w]!=255 and b!=255):
					#right=[h,w]
					right=w
					return right+1
					break
				b=im[h,w]
				h=h-1
			w=w-1

	def bottom():
		b=0
		h=height-1
		while h >= 0:
			w=width-1
			while w >= 0:
				if(im[h,w]!=255 and b!=255):
					#bottom=[w,h]
					bottom=h
					return bottom+1
					break
				b=im[h,w]
				w=w-1
			h=h-1
	return left(),top(),right(),bottom()
	#print("Grid Area: ("+repr(left())+","+repr(top())+"),("+repr(right())+","+repr(bottom())+")")

# im = cv2.imread("aksaraKuno1.png",0)
# im = cv2.imread("3309_GT1.bmp",0)
im = cv2.imread("Table.JPG",0)

# Di Threshold supaya item putih
ret, im= cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Mengambil tinggi & lebar gambar
height, width= im.shape

# Diprint
print(height,width)

# Disave
cv2.imwrite('TH.jpg',im)

left,top,right,bottom=grid(im,height, width)
print("Grid Area: ("+repr(left)+","+repr(top)+"),("+repr(right)+","+repr(bottom)+")")
im=im[top:bottom, left:right]
cv2.imwrite('Grid.jpg',im)
#cv.imwrite(save_path + '%s.png' % file_name, crop)
#image = Image.open(name)
#image = image.resize((30, 30))
#image.save("%s.png" % base)

# Inisialisasi
banyakBlackDiRow = []
h = 0

# Mengambil tinggi & lebar gambar
height, width= im.shape

# Mencari Banyak black pixel per baris
# Sepanjang tinggi gambar
while h < height:
	# Menambah tuple/data di variabel
	banyakBlackDiRow.append(0)
	w=0
	black = 0 
	# Sepanjang lebar gambar
	while w < width:
		# Jika hitam, maka nilai variabel tambah 1
		if im[h,w] == 0:
			black +=1
		w=w+1
	# Total seluruh pixel hitam dalam 1 baris disimpan
	banyakBlackDiRow[h] = black
	h=h+1
"""
# Mencari Banyak black pixel per baris
# Sepanjang tinggi gambar
while h < height:
	# Menambah tuple/data di variabel
	banyakBlackDiRow.append(0)
	w=0
	black = im.item(0) 
	# Total seluruh pixel hitam dalam 1 baris disimpan
	banyakBlackDiRow[h] = black
	h=h+1
"""
# Inisialisasi
banyakBlackDiColumn = []
w=0

# Mencari Banyak black pixel per kolom
# Sepanjang lebar gambar
while w < width:
	# Menambah tuple/data di variabel
	banyakBlackDiColumn.append(0)
	h=0
	black = 0 
	# Sepanjang tinggi gambar
	while h < height:
		# Jika hitam, maka nilai variabel tambah 1
		if im[h,w] == 0:
			black +=1
		h=h+1
	# Total seluruh pixel hitam dalam 1 kolom disimpan
	banyakBlackDiColumn[w] = black
	w=w+1

print (banyakBlackDiRow)
print (banyakBlackDiColumn)

maximumBlackRow = max(banyakBlackDiRow)
maximumBlackColumn = max(banyakBlackDiColumn)
print (maximumBlackRow)
print (maximumBlackColumn)

blackRowPosition = []
blackColumnPosition = []

# Menghapus border horizontal
h=0
while h < height:
	# CHANGE DIS with np.std
	# Jika banyakBlackDiRow = maximumnya 
	if(banyakBlackDiRow[h]==maximumBlackRow):
		print('Baris ke-'+str(h))
		blackRowPosition.append(h)
		w=0
		# Sepanjang lebar gambar
		while w < width:
			# Jika hitam, dijadikan putih
			im[h,w] = 255
			w=w+1
	h=h+1

w=0

# Menghapus border vertikal
w=0
while w < width:
	# CHANGE DIS
	# Jika banyakBlackDiColumn = 80% maximumnya 
	if(banyakBlackDiColumn[w]>(maximumBlackColumn*0.8)):
		print('Kolom ke-'+str(w))
		blackColumnPosition.append(w)
		h=0
		# Sepanjang tinggi gambar
		while h < height:
			# Jika hitam, dijadikan putih
			im[h,w] = 255
			h=h+1
	w=w+1

print(blackRowPosition)
print(blackColumnPosition)

cv2.imwrite('Clean.jpg',im)

numBRP=len(blackRowPosition)
numBCP=len(blackColumnPosition)

for x in xrange(0,len(blackRowPosition)-1):
	for y in xrange(0,len(blackColumnPosition)-1):
		ins=im[blackRowPosition[x]:blackRowPosition[x+1], blackColumnPosition[y]:blackColumnPosition[y+1]]
		cv2.imwrite("Box"+str(x)+str(y)+".jpg",ins)

"""
#picture = Image.open("Grid.JPG")
for h in range(0,height-1):
	if(banyakBlackDiRow[h]==maximumBlackRow):
		print('Baris ke-'+str(h))
		for w in range(0,width-1):
			im[h,w] = [255, 255, 255]
			#picture.putpixel( (h,w), 1)
			#im[h,w]=cv2.bitwise_not(im[h,w])
"""
#picture.save('PIL.jpg')
#cv2.imwrite('clean.jpg',im)
"""
# Menginvert gambar agar komputer bisa membaca gambar dimana 0 = hitam 1 = putih
im = cv2.bitwise_not(im)
# Save
cv2.imwrite('IV.jpg',im)

#Window untuk labeling 
s = [[1,1,1],
     [1,1,1],
     [1,1,1]]
"""


"""
# labeling/ connected commponent analysis untuk melabelisasi tiap huruf/karakter yang nanti akan dicari average heightnya
labeled_array, num_features = sp.label(im,s)
print num_features
print labeled_array

# Untuk Mencari bounding box connected component, yang nanti menghasilkan tupple
props = sk.regionprops(labeled_array)

# mencari Average Height dari tupple bounding box
avgHeight=0
avgWidth=0

i=0
while(i<num_features):
	# Mencari min colom, min row, max row, max col
	(min_row, min_col, max_row, max_col)=props[i].bbox
	i+=1
	# batas atas kurang batas bawah = height
	avgHeight=(avgHeight+(max_row-min_row))
	avgWidth = (avgWidth+(max_col-min_col))

# Mencari rata" nya 
avgHeight = avgHeight/num_features
avgWidth = avgWidth/num_features

print avgHeight
print avgWidth


# Program orang lain untuk smoothing projection 
def moving_average(iterator, length, step=1):
    window = deque(it.islice(iterator, 0, length*step, step))
    total = sum(window)
    yield total / length
    for i in it.islice(iterator, length*step, None, step):
        total -= window.popleft()
        total += i
        window.append(i)
        yield total / length

smoothHist = list(moving_average(banyakBlackDiRow,avgHeight))
print smoothHist

# mencari jumlah puncak/peaks agar bisa menentukan lembah/ baris
h = 0
jumlahPeaks = 0
peaks=[] # Koordinat/baris ke berapa titik puncak

# Mencari Banyak peaks di smoothHistogram
while h < len(smoothHist)-1:
	if(smoothHist[h]>smoothHist[h-1] and smoothHist[h]>smoothHist[h+1]):
		peaks.append(h)
		# menambah nilai ke peaks
		jumlahPeaks+=1
	h=h+1

print jumlahPeaks
print peaks

# mencari HTL = rata" dari selisih peaks /ada dipaper
h=jumlahPeaks-1
HtL=0

while h > 0 :
	HtL = HtL + (peaks[h]-peaks[h-1])
	h=h-1
HtL=HtL/(jumlahPeaks-1)
print HtL

#Membagi teks sesuai baris
splitIm = np.array_split(im,math.trunc(jumlahPeaks))

part=len(splitIm)
print len(splitIm)
i=0
while i<part:
	cv2.imwrite("Part"+str(i)+".jpg",splitIm[i])
	print i
	i+=1

width3=3*avgWidth
n=math.trunc(width/width3)
i=0
while i<part:
	j=0
	while j<n:
#		while k<:

#			splitH = np.array_split(splitIm[i],math.trunc(width/width3))
#			cv2.imwrite("Part"+str(i)+"_"+str(j)+".jpg",splitH[j])

#			k+=1
		j+=1
	i+=1

#print splitIm[0][1,2]
#print list pertama splitIm pixel(1,2)

height=height/jumlahPeaks
width=3*avgWidth
Path="D:\MAD GANGS\Patton\AdaptiveProjection"
input="dummy.png"
k=1
def crop(Path, input, height, width, k, area):
	im = Image.open(input)
	imgwidth, imgheight = im.size
	for i in range(0,imgheight,height):
	    for j in range(0,imgwidth,width):
	        box = (j, i, j+width, i+height)
	        a = im.crop(box)
	        a.save(os.path.join(Path,"PNG IMG-%s.png" % k))
	        print "Success "+str(k)
	        # try:
	        #     o = a.crop(area)
	        #     o.save(os.path.join(Path,"PNG IMG-%s.png" % k))
	        #     print "Success "+str(k)
	        # except:
	        #     pass
	        k +=1


crop(Path, input, height, width, k, 1)
print height*jumlahPeaks
plt.hist(banyakBlackDiRow, bins=163, range=(1,162))
plt.title("Histogram with bins")
fig = plt.gcf()

plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')
plt.show()
"""