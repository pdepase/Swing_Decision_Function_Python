# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:00:12 2024

@author: pdepase
"""

def swing_decision_function(data_set):
    import numpy as np
    import joblib
    
    
    ##COMPLETE CASES ONLY##
    data_set = data_set.dropna(subset=['plate_x'])
    data_set = data_set.dropna(subset=['plate_z'])
    data_set = data_set.dropna(subset=['pfx_x'])
    data_set = data_set.dropna(subset=['pfx_z'])
    data_set = data_set.dropna(subset=['balls'])
    data_set = data_set.dropna(subset=['strikes'])
    
    
    
    ##LOAD AND RUN MODELS TO PREDICT OUTCOMES BASED ON PITCH METRICS##
    called_strike_gam= joblib.load('called_strike_gam.pkl')
    
    data_set['Called_Strike_Prob']= called_strike_gam.predict(data_set[['plate_x','plate_z']])

    
    swing_gam= joblib.load('swing_gam.pkl')
    
    data_set['Swing_Prob']= swing_gam.predict(data_set[['plate_x','plate_z','pfx_x','pfx_z']])


    single_gam= joblib.load('single_gam.pkl')
    
    data_set['Single_Prob']= single_gam.predict(data_set[['plate_x','plate_z','pfx_x','pfx_z']])
    
    
    
    double_gam= joblib.load('double_gam.pkl')
    
    data_set['Double_Prob']= double_gam.predict(data_set[['plate_x','plate_z','pfx_x','pfx_z']])
    
    
    triple_gam= joblib.load('triple_gam.pkl')
    
    data_set['Triple_Prob']= triple_gam.predict(data_set[['plate_x','plate_z','pfx_x','pfx_z']])
    
    
    
    homerun_gam= joblib.load('homerun_gam.pkl')
    
    data_set['HR_Prob']= homerun_gam.predict(data_set[['plate_x','plate_z','pfx_x','pfx_z']])
    
    
    
    out_gam= joblib.load('out_gam.pkl')
    
    data_set['Out_Prob']= out_gam.predict(data_set[['plate_x','plate_z','pfx_x','pfx_z']])
    
    
    
    
    ## CREATE RUNNERS COLUMN##
    
    conditions=[(data_set['on_1b'].isna() & data_set['on_2b'].isna() & data_set['on_3b'].isna()),
                (data_set['on_1b'].notna() & data_set['on_2b'].isna() & data_set['on_3b'].isna()),
                (data_set['on_1b'].isna() & data_set['on_2b'].notna() & data_set['on_3b'].isna()),
                (data_set['on_1b'].isna() & data_set['on_2b'].isna() & data_set['on_3b'].notna()),
                (data_set['on_1b'].notna() & data_set['on_2b'].notna() & data_set['on_3b'].isna()),
                (data_set['on_1b'].notna() & data_set['on_2b'].isna() & data_set['on_3b'].notna()),
                (data_set['on_1b'].isna() & data_set['on_2b'].notna() & data_set['on_3b'].notna()),
                (data_set['on_1b'].notna() & data_set['on_2b'].notna() & data_set['on_3b'].notna())
                ]


    choices= ['000','100', '010', '001','110','101', '011','111' ]
    
    
    data_set['Runners']=np.select(conditions, choices, default='ERROR')
    
    data_set['Runners'] = data_set['Runners'].astype(str)
    
    ##DETERMINE RUN VALUE OF AN OUT## 
    data_set['outs_when_up'] = data_set['outs_when_up'].astype(str)
    
    
    conditions2=[((data_set['Runners']=="000") & (data_set['outs_when_up']=="0")),
                 ((data_set['Runners']=="000") & (data_set['outs_when_up']=="1")),
                 ((data_set['Runners']=="000") & (data_set['outs_when_up']=="2")),
                 ((data_set['Runners']=="100") & (data_set['outs_when_up']=="0")),
                 ((data_set['Runners']=="100") & (data_set['outs_when_up']=="1")),
                 ((data_set['Runners']=="100") & (data_set['outs_when_up']=="2")),
                 ((data_set['Runners']=="010") & (data_set['outs_when_up']=="0")),
                 ((data_set['Runners']=="010") & (data_set['outs_when_up']=="1")),
                 ((data_set['Runners']=="010") & (data_set['outs_when_up']=="2")),
                 ((data_set['Runners']=="001") & (data_set['outs_when_up']=="0")),
                 ((data_set['Runners']=="001") & (data_set['outs_when_up']=="1")),
                 ((data_set['Runners']=="001") & (data_set['outs_when_up']=="2")),
                 ((data_set['Runners']=="110") & (data_set['outs_when_up']=="0")),
                 ((data_set['Runners']=="110") & (data_set['outs_when_up']=="1")),
                 ((data_set['Runners']=="110") & (data_set['outs_when_up']=="2")),
                 ((data_set['Runners']=="101") & (data_set['outs_when_up']=="0")),
                 ((data_set['Runners']=="101") & (data_set['outs_when_up']=="1")),
                 ((data_set['Runners']=="101") & (data_set['outs_when_up']=="2")),
                 ((data_set['Runners']=="011") & (data_set['outs_when_up']=="0")),
                 ((data_set['Runners']=="011") & (data_set['outs_when_up']=="1")),
                 ((data_set['Runners']=="011") & (data_set['outs_when_up']=="2")),
                 ((data_set['Runners']=="111") & (data_set['outs_when_up']=="0")),
                 ((data_set['Runners']=="111") & (data_set['outs_when_up']=="1")),
                 ((data_set['Runners']=="111") & (data_set['outs_when_up']=="2")),
                 ]


    choices2=[-0.25,-0.18,-0.11,-0.38,-0.33,-0.24,-0.44,-0.39,-0.34,-0.42,-0.62,-0.38,-0.55,-0.54,-0.46,-0.55,-0.69,-0.55,-0.61,-0.82,-0.61, -0.67,-0.86,-0.77]




    data_set['RV_Out']=np.select(conditions2, choices2, default='ERROR')


    data_set['RV_Out']=data_set['RV_Out'].astype(float)
    
    
    
    ##DETERMINE RUN VALUE OF A WALK##
    choices3=[0.41,0.28,0.13,0.6,0.43,0.22,0.38,0.27,0.12,0.37,0.24,0.17,0.75,0.63,0.31,0.51,0.39,0.22,0.26,0.2,0.16,1,1,1]


    data_set['RV_Walk']=np.select(conditions2, choices3, default='ERROR')
    
    
    
    data_set['RV_Walk']=data_set['RV_Walk'].astype(float)
    
    
    ##DETERMINE RUN VALUE OF A BALL##
    
    conditions4= [((data_set['balls']==0) & (data_set['strikes']==0)),
                  ((data_set['balls']==1) & (data_set['strikes']==0)),
                  ((data_set['balls']==2) & (data_set['strikes']==0)),
                  ((data_set['balls']==0) & (data_set['strikes']==1)),
                  ((data_set['balls']==1) & (data_set['strikes']==1)),
                  ((data_set['balls']==2) & (data_set['strikes']==1)),
                  ((data_set['balls']==0) & (data_set['strikes']==2)),
                  ((data_set['balls']==1) & (data_set['strikes']==2)),
                  ((data_set['balls']==2) & (data_set['strikes']==2)),
                  (data_set['balls']==3)
                  ]

    choices4=[0.036108717,0.069174515,0.146047634655311,0.0226912526927981,0.0486392876019061,0.118238221335809,0.0373043974898094,0.0500591067994056,0.0874434324885823,data_set['RV_Walk']]
    
    
    
    data_set['Count_Ball_RV']=np.select(conditions4, choices4, default='ERROR')
    
    
    data_set['Count_Ball_RV']=data_set['Count_Ball_RV'].astype(float)
    
    
    
    ##DETERMINE RUN VALUE OF A STRIKE##
    
    conditions5= [((data_set['balls']==0) & (data_set['strikes']==0)),
                  ((data_set['balls']==1) & (data_set['strikes']==0)),
                  ((data_set['balls']==2) & (data_set['strikes']==0)),
                  ((data_set['balls']==3) & (data_set['strikes']==0)),
                  ((data_set['balls']==0) & (data_set['strikes']==1)),
                  ((data_set['balls']==1) & (data_set['strikes']==1)),
                  ((data_set['balls']==2) & (data_set['strikes']==1)),
                  ((data_set['balls']==3) & (data_set['strikes']==1)),
                  (data_set['strikes']==2)
                  ]


    choices5=[-0.0500955001440635,-0.0635129642662081,-0.0840481921616983,-0.1118576054812,-0.0602540389132455,-0.0456408941162342,-0.0442210749187347,-0.0750158637659617,data_set['RV_Out']]


    data_set['Count_Strike_RV']=np.select(conditions5, choices5, default='ERROR')


    data_set['Count_Strike_RV']=data_set['Count_Strike_RV'].astype(float)
    
    
    
    
    ##DETERMINE RUN VALUE OF A SWING##
    data_set['xRV_Swing']= (
        0.883* data_set['Single_Prob']+
        1.238* data_set['Double_Prob']+
        1.558* data_set['Triple_Prob']+
        1.979* data_set['HR_Prob']+
        data_set['RV_Out'] * data_set['Out_Prob']
        )
    
    ##DETERMINE RUN VALUE OF A TAKE##
    data_set['xRV_Take']= (
        data_set['Called_Strike_Prob']* data_set['Count_Strike_RV']+
        data_set['Count_Ball_RV']* (1-data_set['Called_Strike_Prob'])
        )
    
    
    ##DETERMINE WHETHER THE BATTER SHOULD SWING OR TAKE##
    data_set['Correct_Decision']=np.where(data_set['xRV_Swing']<data_set['xRV_Take'], "Take","Swing")


    ##DETERMINE IF THE BATTER MADE THE CORRECT SWING DECISION##
    data_set['IsCorrectDecision']=np.where(((data_set['Correct_Decision']=="Swing") & (data_set['IsSwing']==1))|((data_set['Correct_Decision']=="Take") & (data_set['IsSwing']==0)),1,0)
    
    
    ##DETERMINE RUN VALUE OF THE SWING DECISION AND WEIGHT IT BASED ON HOW OFTEN THE LEAUE MAKES THE SAME DECISION##
    data_set['RV_of_Result']= np.where(data_set['IsSwing']==1,data_set['xRV_Take']*(1-data_set['Swing_Prob']),
                                           np.where(data_set['IsSwing']==0,data_set['xRV_Take']*data_set['Swing_Prob'],"Error"))


    data_set['RV_of_Result']=data_set['RV_of_Result'].astype(float)


    ##GIVE EACH DECISION SCORE A GRADE##
    conditions6=[(data_set['RV_of_Result']>=0.01648372),
                 ((data_set['RV_of_Result']<0.01648372) & (data_set['RV_of_Result']>=0.007272974 )),
                 ((data_set['RV_of_Result']<0.007272974) & (data_set['RV_of_Result']>=0.001374903 )),
                 ((data_set['RV_of_Result']<0.001374903) & (data_set['RV_of_Result']>=-0.01711077  )),
                 (data_set['RV_of_Result']<-0.01711077 )
                 ]


    choices6=["A","B","C","D","F"]


    data_set['Swing_Decision_Grade']=np.select(conditions6, choices6, default='ERROR')


    data_set['Swing_Decision_Grade']=data_set['Swing_Decision_Grade'].astype(str)
    
    
    return data_set











    