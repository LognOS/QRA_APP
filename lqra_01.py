from pickle import TRUE
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import time
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from utils import *
from ml_test_01 import *

st.set_page_config(page_title='QSRA lognos', page_icon='lognos_log_01.png', layout='wide' , initial_sidebar_state= 'collapsed' , menu_items={'Get Help': 'https://vcubo.co/contact','Report a bug': "https://vcubo.co/contact",'About': " Unbiased risk ananlysis. *vcubo*"})
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# MODEL #
## Underlying distriburion (modeling & regression results) / link to private repository pending
df_distrib = pd.DataFrame([['lognormal', 0.1, 0.3, -0.3]], columns = ['type', 'mu', 'sigma', 'shift'])

## Coefficients (modeling & regression results) / link to private repository pending:
df_coef = {'PIONEER':0.2,'COMPLEX':0.3,'DEFINITION':0.5,'CONCURRENCY':0.5,'CONTRACT':0.05,'SOC':0.09,'PROC':0.45,'ENG':0.37,'WEA':0.2,'MGM':0.07,'MIT_ef':1}

## List of variables / dictionary:
df_part_index = ['Pioneer','Complexity','Definition','Concurrency', 'Contract type ',
                               'Social', 'Procurement', 'Labor', 'Weather', 'Management']

risk_dict = {'Social':['SOC', 'SOC_MIT', 'SOC (NM)' ],
        'Procurement':['PROC', 'PROC_MIT', 'PROC (NM)'],
        'Labor':['ENG', 'ENG_MIT', 'LAB (NM)'],
        'Weather':['WEA', 'WEA_MIT', 'WEA (NM)'],
        'Management':['MGM', 'MGM_MIT', 'MGM (NM)']
        }
features_lists = {'PIONEER': ['All', 'YES', 'NO'], 'COMPLEX':['All', '1 - 2', '3 - 4', '5 - 6', '7+'], 'DEFINE':['All', '<20%', '20% - 40%', '40% - 60%', '60% - 80%', '>80%'], 'CONCUR':['All', '>0%', '0% - 20%','20% - 40%', '40% - 60%', '60% - 80%', '>80%'], 'CONTRACT':['All','FIXED', 'OTHER'], 'PROGRESS':['INITIAL', 'Q1', 'Q2', 'Q3', 'Q4']}

# DATA IMPORT #
# db_raw_path = 'https://raw.githubusercontent.com/vcubo/beta_0.1/main/VCDB_221007v3.csv'
db_raw_path = 'https://raw.githubusercontent.com/vcubo/beta_0.1/main/VCDB_230731v5_beta.csv'


#df = import_df(db_raw_path) # main dataframe for general use
# Initial dataframe upload:
st.session_state.df = import_df(db_raw_path) # secondary dataframe for individual project operations


st.write("**PROJECT SETUP**")
with st.form('project_setup'):
    #Initialization of characteristics variables (parameters):
    prf01, prf02, prf03, prf04, prf05, prf06, prf07 = st.columns(7)
    if 'select_PIONEER2' not in st.session_state: st.session_state.select_PIONEER2 = "All"
    if 'select_COMPLEXITY2' not in st.session_state: st.session_state.select_COMPLEXITY2 = "All"
    if 'select_DEFINITION2' not in st.session_state: st.session_state.select_DEFINITION2 = "All"
    if 'select_CONCURRENCY2' not in st.session_state: st.session_state.select_CONCURRENCY2 = "All"
    if 'select_CONTRACT2' not in st.session_state: st.session_state.select_CONTRACT2 = "All"
    if 'select_progress2' not in st.session_state: st.session_state.select_progress2 = "INITIAL"
    if 'hist_xbin_size2' not in st.session_state: st.session_state.hist_xbin_size2 = 0.02
    

    #Generate list of unique values for each parameter in the db. User selection is stored in initialized variables
    with prf01: st.selectbox('PIONEER PLANT',features_lists['PIONEER'], key='select_PIONEER2')
    with prf02: st.selectbox('COMPLEXITY',features_lists['COMPLEX'], key='select_COMPLEXITY2')
    with prf03: st.selectbox('LEVEL OF DEFINITION',features_lists['DEFINE'], key='select_DEFINITION2')
    with prf04: st.selectbox('CONCURRENCY',features_lists['CONCUR'], key='select_CONCURRENCY2')
    with prf05: st.selectbox('CONTRACT TYPE',features_lists['CONTRACT'], key='select_CONTRACT2')
    with prf06: st.selectbox('PROJECT PROGRESS', features_lists['PROGRESS'], key='select_progress2')
    with prf07: st.slider('Histograms bin width', 0.01, 0.1, key='hist_xbin_size2')

    if 'df_pre' not in st.session_state: st.session_state.df_pre = st.session_state.df.copy(deep=True)
    #Applying filters:
    setup_project = st.form_submit_button('SET UP PROJECT')
    if setup_project:
        # generate list of filters applied:
        st.session_state.selection_pro = [st.session_state.select_PIONEER2, st.session_state.select_COMPLEXITY2, st.session_state.select_DEFINITION2, st.session_state.select_CONCURRENCY2, st.session_state.select_CONTRACT2, st.session_state.select_progress2]
        # generate boolean list to apply filters to projects list:
        st.session_state.filter_list2 = filter_gen(st.session_state.selection_pro,st.session_state.df)
        # copy of df with the boolean list filter applied:
        st.session_state.df_pre = st.session_state.df[st.session_state.filter_list2].copy(deep=True)
    
    pr_setup = st.expander("DB DISTRIBUTIONS", expanded=False)
    with pr_setup:
        # Get stastistics and figures from filtered df      
        figures_pre = const_figures(st.session_state.df_pre, st.session_state.df_pre, st.session_state.hist_xbin_size2, df_coef, df_part_index)
        st.session_state.pre_stat = df_stats(st.session_state.df_pre)

        pr01a, pr01b = st.columns(2)
        with pr01a:
            st.write('**TOTAL DEVIATION DISTRIBUTION** (% of deterministic duration)')

            st.plotly_chart(figures_pre[0], use_container_width=True)
            st.caption('Data points: '+str(len(st.session_state.df_pre))+' ')
        with pr01b:
            st.write("**DECOMPOSED DISTRIBUTIONS** (uncertainty and risks effects)")
            st.plotly_chart(figures_pre[1], use_container_width=True)


    pr_analysis = st.expander('PRE-MITIGATION STATISTICS', expanded=False)
    with pr_analysis:
        if 'hist_xbin_size3' not in st.session_state: st.session_state.hist_xbin_size3 = 0.02
        #pr02s1, pr02s2 = st.columns((1,5))
        #with pr02s1:
        st.session_state.figures_p1_fit = fit_distr(st.session_state.df_pre, st.session_state.hist_xbin_size2)
        pr02a, pr02b, pr02c = st.columns((4,1,4))

        with pr02a:
            # st.slider('Fitting curve step length', 0.01, 0.1, key='hist_xbin_size3')
            st.write("**OVERALL STATISTICS:**")
            st.info('Schedule P50 deviation **: '+str(np.round(st.session_state.pre_stat['median'][2]*100,1))+'%**')
            
            st.caption('Uncertainty P50 impact: '+str(np.round((st.session_state.pre_stat['factors'][0]-1)*100,1))+"%. Risks P50 impact: "+str(np.round((st.session_state.pre_stat['factors'][1]-1)*100,1))+"%")

            st.caption('**Lognormal fit** P50: '+str(np.round(fit_probs(st.session_state.figures_p1_fit[5])[0]*100,1))+'%. P80: '+str(np.round(fit_probs(st.session_state.figures_p1_fit[5])[1]*100,1))+'%. Approx. uncertainty impact: '+str(np.round((st.session_state.pre_stat['factors'][0]-1)*100,0))+'%. Approx. risks impact: '+str(np.round((st.session_state.pre_stat['factors'][1]-1)*100,0))+'% (based on '+str(len(st.session_state.df_pre))+' data points)')
           
            st.caption('')

        # with pr02b:
            # st.write('**LOGNORMAL FITING**')
            st.plotly_chart(st.session_state.figures_p1_fit[0], use_container_width=True)

            st.caption('Fitting partitions: '+str(np.round(fit_probs(st.session_state.figures_p1_fit[5])[2],1))+
            ". Sum: "+str(np.round(fit_probs(st.session_state.figures_p1_fit[5])[3],1))+
            ". Totals: "+str(np.round(fit_probs(st.session_state.figures_p1_fit[5])[4],1)))
        # with pr02b:
        #    st.subheader('Lognormal fitting PDF and CDF')
        #    st.plotly_chart(st.session_state.figures_p1_fit[1], use_container_width=True)
        
        with pr02c:
            st.write("**UNCERTAINTY AND RISK DECOMPOSITION**")
            st.plotly_chart(figures_pre[2], use_container_width=True)
            #st.subheader('Lognormal fitting to modeled distribution:')
            #st.plotly_chart(st.session_state.figures_p1_fit[0], use_container_width=True)
        with pr02a:
            # ml_results = predict_features (st.session_state.df_pre, st.session_state.models[0], st.session_state.models[1])

            # features = ['Q1', 'YES', '1 - 2', '<20%', '<0%', 'OTHER']
            # predictions = predict_features(features)
            st.write("**PROJECT RISK QUANTIFICATION**")


var_corr = st.expander('CORRELATION ANALYSIS', expanded=False)
with var_corr:
    st.caption("Estimated total risks impact vs risk type impact" )
    #pr03a, pr03b, pr03c = st.columns((1,2,2))
    #with pr03a:
    #    x3d_sel = st.radio('Risk event type', ['Social','Procurement','Labor', 'Weather', 'Management'])
    #with pr03b:
    #    partials_df_comp = compute_partials(st.session_state.df_pre, df_part_index, df_coef)
    #    sel_dev_corr = sp.stats.pearsonr(partials_df_comp['DEV_EVE'],partials_df_comp[x3d_sel]) #Pearsons' correlation coefficient between composed risk events impact and selected risk type
    #    st.subheader("The Pearson's correlation coefficient between the total impact of risk events and the estimated impact of "+ x3d_sel+ " risk events " +str(np.round(sel_dev_corr[0],2))+".")
    #    st.subheader("The p-value of the correlation coefficient is "+str(np.round(sel_dev_corr[1]*100,2))+"%.")

    pr04a, pr04b, pr04c = st.columns((2,4,4))
    with pr04a:
        x3d_sel = st.radio('Risk event type', ['Social','Procurement','Labor', 'Weather', 'Management'])
        st.session_state.df_pre_2 = st.session_state.df_pre.copy(deep=True)
        st.session_state.partials_df_comp = compute_partials(st.session_state.df_pre_2, df_part_index, df_coef)
        sel_dev_corr = sp.stats.pearsonr(st.session_state.partials_df_comp['DEV_EVE'],st.session_state.partials_df_comp[x3d_sel]) #Pearsons' correlation coefficient between composed risk events impact and selected risk type
        corr_summary = classify_correlation(sel_dev_corr)
        st.caption("The Pearson's correlation coefficient between the total impact of risk events and the estimated impact of "+ x3d_sel+ " risk events is " +str(np.round(sel_dev_corr[0],2))+". The p-value of the correlation coefficient is "+str(np.round(sel_dev_corr[1]*100,2))+"%. ")
        st.caption(corr_summary)

    with pr04b:
        title_scatter1="DEVIATION  vs EVENTS (by type)"
        st.plotly_chart(scatter_hist(st.session_state.partials_df_comp,risk_dict[x3d_sel][2], title_scatter1), use_container_width=True)
    with pr04c:
        title_scatter2="TOTAL VS SELECTED TYPE DEVIATIONS"
        st.plotly_chart(scatter_hist(st.session_state.partials_df_comp,x3d_sel, title_scatter2), use_container_width=True)
#st.plotly_chart(scatter_3dim (st.session_state.df_pre, risk_dict[x3d_sel][0], risk_dict[x3d_sel][1], 'DEV_EVE', 'DEV_EVE', 'DEV_TOT'))

# with tab3:
st.write("**PROJECT RISK QUANTIFICATION**")
risk_mitigate = st.expander('MITIGATION AND TUNING', expanded=False)
# Generates df and figures in mitigation page in case a project wasn't set in the previus step
if 'hist_xbin_size2' not in st.session_state:
    st.session_state.hist_xbin_size2=0.05
    st.session_state.hist_xbin_size3=0.05
# if 'df_pos' not in st.session_state:
#     selection_gen = 5*['All']
#     st.session_state.df_pre = df[filter_gen(selection_gen,df)].copy(deep=True)
#     st.session_state.pre_stat = df_stats(st.session_state.df_pre)
#     figures_pre = const_figures(st.session_state.df_pre, st.session_state.df_pre, st.session_state.hist_xbin_size2, df_coef, df_part_index)
#     figures_p1_fit = fit_distr(st.session_state.df_pre, st.session_state.hist_xbin_size3)
#     st.session_state.figures_p1_fit = figures_p1_fit

st.session_state.df_pos = st.session_state.df_pre.copy(deep=True)

if 'soc_mit' not in st.session_state: st.session_state.soc_mit = 0.0
if 'proc_mit' not in st.session_state: st.session_state.proc_mit = 0.0
if 'eng_mit' not in st.session_state: st.session_state.eng_mit = 0.0
if 'wea_mit' not in st.session_state: st.session_state.wea_mit = 0.0
if 'mgm_mit' not in st.session_state: st.session_state.mgm_mit = 0.0

if 'pio_adj' not in st.session_state: st.session_state.pio_adj = 0.0
if 'comp_adj' not in st.session_state: st.session_state.comp_adj = 0.0
if 'def_adj' not in st.session_state: st.session_state.def_adj = 0.0
if 'conc_adj' not in st.session_state: st.session_state.conc_adj = 0.0
if 'cont_adj' not in st.session_state: st.session_state.cont_adj = 0.0

with risk_mitigate:
    
    #form = st.form('Mitigation sliders', clear_on_submit=True)
    pr05, pr05a, pr05b, pr05c,pr05c2, pr05d = st.columns((1,4,1,4,1,10))
    #with form:
    with pr05a:
        st.write("**RISK ADJUSTMENT**")
        st.slider('Social risks',0.0, 1.0, step=0.1, help="Adjust mitigation factor for the risk types", key='soc_mit')
        st.slider('Procurement risks',0.0, 1.0, step=0.1, key='proc_mit')
        st.slider('Labor risks',0.0, 1.0, step=0.1, key='eng_mit')
        st.slider('Weather risks',0.0, 1.0, step=0.1, key='wea_mit')
        st.slider('Management risks',0.0, 1.0, step=0.1, key='mgm_mit')
        st.session_state.mitigation = [st.session_state.soc_mit, st.session_state.proc_mit, st.session_state.eng_mit, st.session_state.wea_mit, st.session_state.mgm_mit, st.session_state.pio_adj, st.session_state.comp_adj, st.session_state.def_adj, st.session_state.conc_adj, st.session_state.cont_adj]
        #reset_mit = form.form_submit_button('Reset')
        #if reset_mit:
    #st.write(pos_stat)

    with pr05c:
        st.write("**UNCERTAINITY TUNING**")
        st.slider('Pioneer',0.0, 1.0, step=0.1, help="Adjust systemic uncertainity", key='pio_adj')
        st.slider('Complexity',0.0, 1.0, step=0.1, key='comp_adj')
        st.slider('Level of definition',0.0, 1.0, step=0.1, key='def_adj')
        st.slider('Concurrency',0.0, 1.0, step=0.1, key='conc_adj')
        st.slider('Contract type',0.0, 1.0, step=0.1, key='cont_adj')
        st.session_state.mitigation = [st.session_state.soc_mit, st.session_state.proc_mit, st.session_state.eng_mit, st.session_state.wea_mit, st.session_state.mgm_mit, st.session_state.pio_adj, st.session_state.comp_adj, st.session_state.def_adj, st.session_state.conc_adj, st.session_state.cont_adj]
    # st.code(st.session_state.mitigation)
    # st.dataframe(st.session_state.df_pre)
    # st.dataframe(st.session_state.df_pos)
    st.session_state.df_pos = update_impact(st.session_state.df_pos, st.session_state.df_pre, st.session_state.mitigation, df_coef)[0]
    st.session_state.figures_pos = const_figures(st.session_state.df_pre, st.session_state.df_pos, st.session_state.hist_xbin_size3, df_coef, df_part_index)
    st.session_state.figures_pos_fit = fit_distr(st.session_state.df_pos, st.session_state.hist_xbin_size3)
    st.session_state.pos_stat = df_stats(st.session_state.df_pos)  

    with pr05d:
        st.plotly_chart(st.session_state.figures_pos[2], use_container_width=True)
        #st.subheader("With a lognormal fit over delays' distribution of the "+str(len(st.session_state.df_pre))+" projects selected.")
        #st.subheader('Distribution mean: '+str(np.round(st.session_state.pos_stat['means'][2]*100,2))+'%(fit) vs '+str(np.round(st.session_state.pre_stat['means'][2]*100,1))+'%(fit)')
        st.info('Distribution median **(P50): '+str(np.round(st.session_state.pos_stat['median'][2]*100,1))+'%**  vs '+str(np.round(st.session_state.pre_stat['median'][2]*100,1))+'% pre-mitigation')

        st.caption('Uncertainty P50 impact: '+str(np.round((st.session_state.pos_stat['factors'][0]-1)*100,1))+"%. Risks P50 impact: "+str(np.round((st.session_state.pos_stat['factors'][1]-1)*100,1))+"%")

        # st.caption('**Lognormal fit** P50: '+str(np.round(fit_probs(st.session_state.figures_p1_fit[5])[0]*100,1))+'%. P80: '+str(np.round(fit_probs(st.session_state.figures_p1_fit[5])[1]*100,1))+'%. Approx. uncertainty impact: '+str(np.round((st.session_state.pos_stat['factors'][0]-1)*100,0))+'%. Approx. risks impact: '+str(np.round((st.session_state.pos_stat['factors'][1]-1)*100,0))+'% (based on '+str(len(st.session_state.df_pos))+' data points)')

        # df_polar =pd.DataFrame.from_dict({'Risk type':['Social', 'Procurement', 'Engineering', 'Weather', 'Management'], '% Mitigation':st.session_state.mitigation[:5] })
        # polar_mit = px.line_polar(df_polar, r="% Mitigation", theta="Risk type", line_close=True, #color="% Mitigation",
        #            color_discrete_sequence=px.colors.sequential.Rainbow,
        #            template="plotly_dark")
        # mitigated_plot = scatter_hist(compute_partials(st.session_state.df_pos, df_part_index, df_coef),risk_dict[x3d_sel][2],"Mitigation")
        # st.plotly_chart(polar_mit)



show_charts = st.expander('POS_MITIGATION SUMMARY', expanded=False)
x = np.linspace(0,1,int(1/st.session_state.hist_xbin_size3))

mit_dist_chart = st.session_state.figures_p1_fit[1]
with show_charts:
    pr08a, pr08b, pr08c = st.columns((4,1,4))

    with pr08b:
        upd_mit_chart = st.button('COMPARE', help="Add post-mitigation probability distribution",)
        #cln_mit_chart = st.button('Clean ')
        if upd_mit_chart:
            mit_dist_chart.add_scatter(y = st.session_state.figures_pos_fit[4], x = x, name = 'Mitigated pdf')
        #if cln_mit_chart:
        #    mit_dist_chart = st.session_state.figures_p1_fit[1]
    with pr08a:
        st.text('SAMPLE LOGNORMAL FIT')
        st.plotly_chart(st.session_state.figures_p1_fit[0], use_container_width=True)
    with pr08c:


        st.text('FIT DISTRIBUTION UPDATE')
        st.plotly_chart(mit_dist_chart, use_container_width=True)

    with pr08a: st.plotly_chart(st.session_state.figures_pos[0], use_container_width=True)
    with pr08c: st.plotly_chart(st.session_state.figures_pos[1], use_container_width=True)
