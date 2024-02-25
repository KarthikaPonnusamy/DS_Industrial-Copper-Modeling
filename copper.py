import streamlit as st
import streamlit_option_menu
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeRegressor
import re
from PIL import Image

#Page config
icon = Image.open("copperprice.png")
st.set_page_config(page_title= "DS_Industrial Copper Modeling",
                   page_icon= icon,
                   layout= "wide",
                   initial_sidebar_state= "expanded"
                   )
# SETTING-UP BACKGROUND IMAGE
def setting_bg():
    st.markdown(f""" <style>.stApp {{
                        background:url("https://e1.pxfuel.com/desktop-wallpaper/624/979/desktop-wallpaper-soft-yellow-backgrounds-backgrounds-background-colour-soft.jpg");
                        background-size: cover}}
                     </style>""", unsafe_allow_html=True)


setting_bg()

st.write("""
<div style='text-align:center'>
    <h1 style='color:#0A0A0A;'>Industrial Copper Modeling Application</h1>
</div>
""", unsafe_allow_html=True)



tab1, tab2 = st.tabs(["PREDICT SELLING PRICE","PREDICT STATUS"])


with tab1:
        status_opt = ['Won','Draft','Lost','No lost for AM','Offerable','Offered','Revised','To be approved','Wonderful']
        item_type_opt = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        country_opt = [28., 25., 30., 32., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        application_opt = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67.,
                           79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
        product = ['611993', '611728', '628112', '628117', '628377', '640400', '640405', '640665','611993', '929423819',
               '1282007633', '1332077137', '164141591', '164336407',
               '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
               '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
               '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
        
        with st.form('copper_form'):
                col1,col2,col3 = st.columns([4,1,4])

                with col1:
                        st.write(' ')
                        status = st.selectbox("Status",status_opt)
                        item = st.selectbox("Item",item_type_opt)
                        country = st.selectbox("Country",country_opt)
                        application = st.selectbox("Application",application_opt)
                        product_ref = st.selectbox("Prodcuct Reference",product)

                with col3:
                        st.write(' ')
                        quantity_tons = st.text_input('Enter Quantity Tons (Min:611728 & Max:1722207579)')
                        thickness = st.text_input('Enter thickness (Min:0.18 & Max:400)')
                        width = st.text_input('Enter width(Min:1, Max:2990)')
                        customer = st.text_input('customer ID (Min:12458, Max:30408185)')
                        submit_button = st.form_submit_button(label='Predict Selling Price')
                        st.markdown('''
                        ''', unsafe_allow_html=True)

                flag = 0
                pattern = '^(?:\d+|\d*\.\d+)$'
                for i in [quantity_tons,thickness,width,customer]:
                        if re.match(pattern,i):
                                pass
                        else:
                                flag=1
                                break


        if submit_button and flag==1:
                if len(i)==0:
                        st.write('please enter a valid number space not allowed')
                else:
                        st.write('you have entered an invalid value: ', i)

                
        if submit_button and flag==0:

                import pickle

                with open(r'model.pkl','rb') as file:
                    loaded_model = pickle.load(file)    
                with open(r'scaler.pkl','rb') as file:
                    scaler_model = pickle.load(file)
                with open(r'item.pkl','rb') as file:
                    item_model = pickle.load(file)
                with open(r'status.pkl','rb') as file:
                    status_model = pickle.load(file)


                new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item,status]])
                new_sample_ohe = item_model.transform(new_sample[:, [7]]).toarray()
                new_sample_be = status_model.transform(new_sample[:, [8]]).toarray()
                new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
                new_sample1 = scaler_model.transform(new_sample)
                new_pred = loaded_model.predict(new_sample1)[0]
                st.write('## :green[Predicted selling price:] ', np.exp(new_pred))


with tab2: 
    
        with st.form("my_form1"):
            col1,col2,col3=st.columns([5,1,5])
            with col1:
                cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                cselling = st.text_input("Selling Price (Min:1, Max:100001015)") 
              
            with col3:    
                st.write(' ')
                citem_type = st.selectbox("Item Type", item_type_opt,key=21)
                ccountry = st.selectbox("Country", sorted(country_opt),key=31)
                capplication = st.selectbox("Application", sorted(application_opt),key=41)  
                cproduct_ref = st.selectbox("Product Reference", product,key=51)           
                csubmit_button = st.form_submit_button(label="PREDICT STATUS")
    
            cflag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:             
                if re.match(pattern, k):
                    pass
                else:                    
                    cflag=1  
                    break
            
        if csubmit_button and cflag==1:
            if len(k)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",k)  
             
        if csubmit_button and cflag==0:
                import pickle
                with open(r"cmodel.pkl", 'rb') as file:
                        cloaded_model = pickle.load(file)

                with open(r'cscaler.pkl', 'rb') as f:
                        cscaler_loaded = pickle.load(f)

                with open(r"ct.pkl", 'rb') as f:
                        ct_loaded = pickle.load(f)




                new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication,
                                        np.log(float(cthickness)), float(cwidth), ccountry, int(ccustomer), int(product_ref),
                                        citem_type]])
                new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
                new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_ohe), axis=1)
                new_sample = cscaler_loaded.transform(new_sample)
                new_pred = cloaded_model.predict(new_sample)
                #st.write(new_pred)
                if new_pred.any()==1:
                        st.write('## :green[The Status is Won] ')
                else:
                        st.write('## :red[The status is Lost] ')
                
