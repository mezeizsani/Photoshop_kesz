from tkinter import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import os


def ok():
    fajlnev = e1.get()
    futasi_hely = os.getcwd()
    kep = os.path.join(futasi_hely, fajlnev)
    if os.path.isfile(kep):
        l2.config(text="Válasszon az alábbi funkciók közül:")
        return kep
    else:
        l2.config(text="A fájl nem létezik, vagy hibás a fájlnév!")
    


def eredeti():
    kep1=ok()
    img = cv2.imread(kep1) 
    # kép megjelenítése
    cv2.imshow('eredeti_kep', img)


#########################################   negalas   #################################

def negal():
    
    kep1=ok()
    # kép beolvasása
    img = cv2.imread(kep1)

    start_time = time.perf_counter()
    # kép negálása
    neg_img = cv2.bitwise_not(img)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # eredmény megjelenítése
    cv2.imshow("Negalt_kep", neg_img)


    print(f"A negálás futási ideje: {elapsed_time:.6f} másodperc")

#########################################   gamma   #################################
   
def gamma():
    
    kep1=ok()
    # kép beolvasása
    img = cv2.imread(kep1)
    start_time = time.perf_counter()
    # gamma érték meghatározása
    gamma = 1.5
    # LUT tábla előállítása
    lut = np.array([((i/255.0)**(1.0/gamma))*255 for i in range(256)]).astype("uint8")
    # LUT alkalmazása a képre
    result = cv2.LUT(img, lut)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # eredmény megjelenítése
    cv2.imshow("Gamma_transzformacio", result)


    print(f"A gamma transzformáció futási ideje: {elapsed_time:.6f} másodperc")

#########################################   log   #################################

def log():
    
    kep1=ok()
    # kép betöltése
    img = cv2.imread(kep1)
    start_time = time.perf_counter()
    # szélességi dinamikatartományhoz igazított állandó c kiszámítása
    c = 255 / np.log(1 + np.max(img))
    # logaritmikus transzformáció
    log_transformed = c * (np.log(img.clip(1)))
    log_transformed = np.uint8(log_transformed)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # eredmény megjelenítése
    cv2.imshow('Log_transzformacio', log_transformed)


    print(f"A logaritmikus transzformáció futási ideje: {elapsed_time:.6f} másodperc")

#########################################   szürkítés   #################################

def gray():

    kep1=ok()
    # Kép beolvasása
    img = cv2.imread(kep1)
    start_time = time.perf_counter()
    # Szürkeárnyalatos kép készítése
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # Eredmény megjelenítése
    cv2.imshow("Szurkites", gray_img)


    print(f"A szürkítés futási ideje: {elapsed_time:.6f} másodperc")
    
   
#########################################   hisztogram   #################################

def histogram():

    kep1=ok()
    # Kép betöltése
    img = cv2.imread(kep1, 0)
    start_time = time.perf_counter()
    # Hisztogram készítése
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # Hisztogram ábrázolása
    plt.hist(img.flatten(), 256, [0,256])
    plt.show()


    print(f"A hisztogram készítésének futási ideje: {elapsed_time:.6f} másodperc")

#########################################   hisztogram kiegyenlites  #################################

def histeq():

    kep1=ok()
    # kép betöltése
    img = cv2.imread(kep1, 0)
    start_time = time.perf_counter()
    # hisztogram kiegyenlítése
    equ = cv2.equalizeHist(img)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # kép megjelenítése
    cv2.imshow('Original', img)
    cv2.imshow('Equalized', equ)


    print(f"A hisztogram kiegyenlítés futási ideje: {elapsed_time:.6f} másodperc")

#########################################   atlagolo szuro   #################################

def atlagolo():

    kep1=ok()
    img = cv2.imread(kep1)
    start_time = time.perf_counter()
    # Az átlagoló szűrő (Box szűrő) alkalmazása
    avg_img = cv2.blur(img, (7, 7))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # Eredmény megjelenítése
    cv2.imshow("Atlagolt_kep", avg_img)


    print(f"Az átlagoló szűrő futási ideje: {elapsed_time:.6f} másodperc")

#########################################   gaussian  #################################

def gauss():

    kep1=ok()
    # kép betöltése
    img = cv2.imread(kep1)
    start_time = time.perf_counter()
    # Gauss szűrő alkalmazása
    gaussian = cv2.GaussianBlur(img, (5,5), 0)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # képek megjelenítése
    cv2.imshow('Gauss_szuro', gaussian)


    print(f"A Gauss szűrő futási ideje: {elapsed_time:.6f} másodperc")

#########################################   sobel   #################################

def sobel():

    kep1=ok()
    # Kép betöltése szürkeárnyalatos képként
    img = cv2.imread(kep1, 0)
    start_time = time.perf_counter()
    # Sobel éldetektor x irányban
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)

    # Sobel éldetektor y irányban
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

    # Teljes gradiens
    sobel = np.sqrt(sobelx**2 + sobely**2)

    # Élek küszöbölése
    sobel = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)[1]
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # Eredmény megjelenítése
    cv2.imshow('Sobel', sobel)


    print(f"A Sobel éldetektor futási ideje: {elapsed_time:.6f} másodperc")


#########################################   laplace eldetektor   #################################

def laplace():

    kep1=ok()
    # betöltjük a képet
    img = cv2.imread(kep1,0)
    start_time = time.perf_counter()
    # alkalmazzuk a Laplace éldetektort
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # megjelenítjük az eredményt
    cv2.imshow('Laplace',laplacian)


    print(f"A Laplace éldetektor futási ideje: {elapsed_time:.6f} másodperc")


#########################################   jellemzopontok detektalasa   #################################

def jell():

    kep1=ok()
    # Kép beolvasása
    img = cv2.imread(kep1)
    start_time = time.perf_counter()
    # Szürkeárnyalatosra konvertálás
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Harris sarokdetektor használata
    dst = cv2.cornerHarris(gray, 3 , 5, 0.04)

    # Jellemzőpontok kiválogatása
    kp = np.argwhere(dst > 0.01 * dst.max())

    # Jellemzőpontok kirajzolása a képre
    for point in kp:
        cv2.circle(img, tuple(point[::-1]), 3, (0, 255, 0), -1)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # Kép megjelenítése
    cv2.imshow('Jellemzopontok', img)


    print(f"A jellemzőpontok detektálásának futási ideje: {elapsed_time:.6f} másodperc")


#########################################   bezar   #################################

def bezar():
 
    cv2.destroyAllWindows()
    plt.close()
    mw.destroy()
    print("A program bezarva!")



#########################################   main window   #################################

mw=Tk()

mw.title("Photoshop_Mezei_Zsanett_JI0W1T")
mw.config(background="ghostwhite",width=400, height=500)

l1=Label(mw, text="Adja meg a kép nevét!\n         (Pl: kep.jpg, kep2.png stb...)", fg="darkorchid4", bg="ghostwhite")
l1.place(x=5, y=50)

l2=Label(mw, text="", fg="violetred4", bg="ghostwhite")
l2.place(x=33, y=88)

e1=Entry(mw, bg="thistle3")
e1.place(x=200, y=60)

b1=Button(mw, text="OK", bg="green4", fg="white", command=ok)
b1.place(x=345, y=56)

b2=Button(mw, text="EXIT", bg="violetred4", fg="white", command=bezar)
b2.place(x=345, y=450)

b3=Button(mw, text="Negálás", bg="mediumorchid4", fg="white",activebackground="orchid3",command=negal)
b3.place(x=30, y=150)

b4=Button(mw, text="Eredeti kép", bg="mediumorchid4", fg="white",activebackground="orchid3",command=eredeti)
b4.place(x=30, y=120)

b5=Button(mw, text="Gamma transzformáció", bg="mediumorchid4", fg="white",activebackground="orchid3",command=gamma)
b5.place(x=30, y=180)

b6=Button(mw, text="Logaritmikus transzformáció", bg="mediumorchid4", fg="white",activebackground="orchid3",command=log)
b6.place(x=30, y=210)

b7=Button(mw, text="Szürkítés", bg="mediumorchid4", fg="white",activebackground="orchid3",command=gray)
b7.place(x=30, y=240)

b8=Button(mw, text="Hisztogram készítés", bg="mediumorchid4", fg="white",activebackground="orchid3",command=histogram)
b8.place(x=30, y=270)

b9=Button(mw, text="Hisztogram kiegyenlítés", bg="mediumorchid4", fg="white",activebackground="orchid3",command=histeq)
b9.place(x=30, y=300)

b10=Button(mw, text="Átlagoló szűrő", bg="mediumorchid4", fg="white",activebackground="orchid3",command=atlagolo)
b10.place(x=30, y=330)

b11=Button(mw, text="Gauss szűrő", bg="mediumorchid4", fg="white",activebackground="orchid3",command=gauss)
b11.place(x=30, y=360)

b12=Button(mw, text="Sobel éldetektor", bg="mediumorchid4", fg="white",activebackground="orchid3",command=sobel)
b12.place(x=30, y=390)

b13=Button(mw, text="Laplace éldetektor", bg="mediumorchid4", fg="white",activebackground="orchid3",command=laplace)
b13.place(x=30, y=420)

b14=Button(mw, text="Jellemzőpontok detektálása", bg="mediumorchid4", fg="white",activebackground="orchid3",command=jell)
b14.place(x=30, y=450)

mw.mainloop()
