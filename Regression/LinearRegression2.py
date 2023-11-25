import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import easygui as eg
from sklearn import linear_model

df = pd.read_csv("Regression/Datas/homeprices.csv", delimiter=';')
print(df)
print(df.columns)
plt.xlabel('area (sqr ft)')
plt.ylabel('price ($)')
plt.scatter(df.area, df.price, color='red', marker='+')
#plt.show() #burda zaten hangi yontemi kullanacagini ogrenebilirsin bu basitce veriyi gorsellestirmek.
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price) #burda linear regresyon modelini egitiyoruz.
#suan fiyatlari tahmin etmeye haziriz
print(reg.predict([[3300]])) #[628715.75342466]
# Regresyon katsayıları mesaj kutusu ile ekrana yazdırılıyor
coef = reg.coef_
print(str(coef)) # [135.78767123]
intcp= reg.intercept_
print(str(intcp)) #180616.43835616432
value = 135.78767123*3300 + 180616.43 # burda intercept ve coefficient i yerlerine koyup hesaplama yaptim
print(value) #628715.745059 
#simdi bu bilgileri kullanarak sadece area kolonunu iceren csv dosyasini okuyup her biri icin tahmin yapacagiz.
d= pd.read_csv("Regression/Datas/areas.csv", delimiter=';')
print(d.columns)
d.columns = ['area']
p = reg.predict(d)
d['prices'] = p
print(d) # cikti su andan itibaren verilen butun verileri kullanarak bu verileri tahmin etti. ve cikti iki sutunlu oldu.
d.to_csv("Regression/Datas/prediction.csv", index=False) # csv dosyasini export ediyoruz.

#------------------------------------------------
#simdi ilk veriye cizgi cekicez tahmine gore
plt.xlabel('area (sqr ft)')
plt.ylabel('price ($)')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
#plt.show()
predicded = pd.read_csv('Regression/Datas/prediction.csv')
#bu da tahmin edilenlerin grafigi, butun noktalar tanimlanan cizginin uzerinde olacak haliyle
plt.xlabel('area (sqr ft)')
plt.ylabel('price ($)')
plt.scatter(predicded.area, predicded.prices, color='red', marker='+')
plt.plot(predicded.area, reg.predict(predicded[['area']]), color='blue')
#plt.show()
#-------------------Linear regression with multiple veriables----------------
df  = pd.read_csv('Regression/Datas/multihomeprices.csv', delimiter=';')
print(df)
import math
median_bedrooms = math.floor(df.bedrooms.median())
print(median_bedrooms) # simdi bu method integer olarak hesaplamayacaktir. ama bize integer lazim. bu yuzden floor kullandik.
df.bedrooms = df.bedrooms.fillna(median_bedrooms) # simdi kolondaki butun bos degerleri medyanla doldurduk.
print(df)
#simdi modeli egitmeye haziriz. 
reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']], df.price)
print(reg.coef_) # bunlar katsayilar
print(reg.intercept_) # bu da sabit
print(reg.predict([[3000,3,40]])) #444400
value = 3000*137.25 + 3*-26025 + 40* -6825 +383725
print(value) # 444400

