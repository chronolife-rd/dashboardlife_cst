import streamlit as st

def run():
    config()
    # hide_streamlit_menu()
    body()
    footer()
    
def config():
    st.set_page_config("Smart Textile", layout="wide")

def hide_streamlit_menu():
    hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
    </style>
    
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    
def body():
    
    st.markdown("""
                <style>               
                    .intro_section {
                        background-image: url('data:image/png;base64,""" + st.session_state.background_wave + """');
                        background-size: cover;
                        margin-top: -100px;
                        margin-bottom: -100px;
                        padding-top: 0px;
                        padding-bottom: 25%;
                        /* border-radius: 30px; */
                        height: 250px;
                        width: 120%;
                        margin-left: -10%;
                    }
                    
                    .intro_style {
                        color: white;
                        font-size: 2em;
                    }
                    
                    .intro_logo {
                        width: 20%;
                        margin-top: 20px;
                        margin-left: 4%;
                        }
                    
                    /* User icon section */
                    .user_icon_section {
                        text-align: center;
                        }
                    
                    /* User icon */
                    .user_icon {
                        width: 50%;
                        }
                    
                    .overview_section {
                        font-size: 24px;
                        background-color: white;
                        color: #3E738D;
                        border: solid darkgrey 0px;
                        border-radius: 30px;
                        padding: 15px;
                        margin-bottom: 20px;
                        }
                    
                    .overview_sub_section_smart_textile {
                        margin-top: -20px;
                        }
                    
                    .main_title {
                        margin-top: 50px;
                        }
                    
                    .main_title p {
                        font-size: 40px;
                        font-weight: bold;
                        }
                    
                    .second_title {
                        margin-top: 10px;
                        }
                    
                    .second_title p {
                        font-size: 30px;
                        }
                    
                    .data_collection {
                        font-size: 18px;
                        color: #3E738D;
                        margin-top: 20px;
                        }
                    
                    .collect_duration {
                        font-size: 18px;
                        font-weight: bold;
                        }
                    
                    .overview_health_section {
                        font-size: 24px;
                        background-color: white;
                        color: #3E738D;
                        border: solid darkgrey 0px;
                        border-radius: 30px;
                        padding: 25px 15px 15px 25px;
                        margin-bottom: 20px;
                        height: 370px;
                        }
                    
                    .health_section {
                        font-size: 24px;
                        background-color: white;
                        color: #3E738D;
                        border: solid darkgrey 0px;
                        border-radius: 30px;
                        padding: 25px 15px 15px 25px;
                        margin-bottom: 20px;
                        height: 300px;
                        }
                    
                    .chronolife_icon {
                        width: 150px;
                        padding-right: 50px;
                        margin-top: 50px;
                        }                    
                    .icon {
                        float: left;
                        width: 40px;
                        margin-right: 20px;
                        }
                    
                    .miniicon {
                        width: 20px;
                        }
                    
                    .coloricon {
                        float: left;
                        width: 10px;
                        margin-right: 5px;
                        margin-top: 7px;
                        }
        
                    .indicator_name {
                        color: #3E738D;
                        font-size: 1.2em;
                        font-weight: bold;
                        }
                    
                    .indicator_value {
                        font-weight: bold;
                        }
        
                    .steps {
                        color: #3E738D;
                        }
                    
                    .steps_number {
                        font-size: 30px;
                        font-weight: bold;
                        margin-top: -8px;
                        }                    
                    .donut { 
                        width: 200px;
                        margin-left: -80px;
                        margin-top: 0px;
                        }

                    /* form section */
                    .epcbefy1 {
                        background-color: white;
                        border-radius: 30px;
                        padding: 30px;
                        border-color: white;
                        }
                    
                    .css-1mv0avt {
                        background-color: white;
                        border-radius: 30px;
                        padding: 30px;
                        border-color: white;
                    
                    }

                    .css-115gedg {
                        # background-color: white;
                        border-radius: 30px;
                        padding: 30px;
                        border-color: white;

                    }
                    
                    # .css-1xtd0iy.e1f1d6gn2 {
                    #     background-color: white;
                    #     border-radius: 30px;
                    #     padding: 30px;
                    #     border-color: white;

                    # }

                   
     


                    
                    /* form label */
                    .effi0qh3 {
                        color: #3F738D;
                        }
                    
                    /* form label */
                    .effi0qh3 p {
                        font-weight: bold;
                        }
                    
                    /* form submit button */
                    .edgvbvh5 {
                        background-color: #3F738D; 
                        border-color: #3F738D;;
                        color: white;
                        border-radius: 20px;
                        }
                    
                    .edgvbvh5:hover {
                        background-color: #5590ad; /* bleu clair => #5590ad; orange clair => #FABB6E;*/
                        border-color: #5590ad; 
                        color: white;
                        }
                    
                    .edgvbvh5:focus {
                        box-shadow: 1px 1px 3px 3px #5590ad !important;
                        border-color: #5590ad !important;
                    }
                    
                    .edgvbvh5 p {
                        color: white;
                        }
                    
                    /* button */
                    .edgvbvh10 {
                        background-color: #3F738D; /* bleu => #3F738D; orange => #F7931E; bleu clair => #5590ad; orange clair => #FABB6E;*/
                        color: white;
                        width: auto !important;
                        border-radius: 30px !important;
                        }
                    
                    .edgvbvh10:hover {
                        background-color: #5590ad; /* bleu clair => #5590ad; orange clair => #FABB6E;*/
                        border-color: #5590ad; 
                        color: white;
                        }
                    
                    .edgvbvh10 p {
                        color: white;
                        }
                    
                    .edgvbvh10 span {
                        color: white;
                        }
                    
                    /* menu button */
                    /* .menu {
                        background-color: #3F738D !important;
                        border-radius: 20px;
                        height: 60px;
                        padding-top: 8px;
                        text-align: center;
                        width: 100%;
                        }
                    */
                    
                    .menu a {
                         /* background-color: #3F738D !important; */
                        text-decoration: none !important;
                        }
                    
                    .menu a span {
                        color: #3F738D;
                        font-size: 20px;
                        font-weight: bold;
                        }
                    
                    .menu a span:hover {
                        color: #CEDFE8 !important;
                        border-color: #5590ad;
                        color: white;
                        }
                    
                    /* buble in front of titles */
                    .e16nr0p32 svg {
                        display: none;
                        }
                    
                    /* duration figure */
                    .main-svg {
                        padding-left: 50px !important;
                        padding-right: 50px !important;
                        border-radius: 30px !important;
                        }
                    
                    /* Display text in orange */
                    .orange_text {
                        color: #F7931E;
                        }
                    
                    /* 1st column of health indicators section */
                    .col1_indicators {
                        background-color: white;
                        border-radius: 30px;
                        padding: 25px;
                        height: 300px;
                        margin-bottom: 20px;
                        }
                    
                    /* Health indicator value */
                    .indicator_value {
                        font-weight: bold;
                        }
                    
                    /* Main Health indicator value */
                    .indicator_main_value {
                        font-size: 20px;
                        font-weight: bold;
                        }
                    
                    /* Definitions section */
                    .definitions_section {
                        font-size: 24px;
                        background-color: white;
                        color: #3E738D;
                        border: solid darkgrey 0px;
                        border-radius: 30px;
                        padding: 15px;
                        margin-bottom: 20px;
                        }
                    
                    
                    /* Definitions icons */
                    .definitions_miniicon {
                        float: left;
                        width: 20px;
                        margin-right: 8px;
                        }
                    
                    /* expander */
                    .streamlit-expander {
                        border-radius: 30px;
                        background-color: white;
                        }
                    
                    /*expander title */
                    .streamlit-expander .css-9plt3t p {
                        font-weight: bold;
                        }
                    
                    /* enduser list */
                    /*.enduser {
                        background-color: #5590ad;
                        color: white;
                        padding-left: 10px;
                        padding-right: 10px;
                        border-radius: 15px;
                        }*/
                    
                    /* sessions table */
                    .e1tzin5v3 .e19lei0e0 {
                        text-align: center;
                        }
                    
                    /* Language Label */
                    .language_label {
                        margin-bottom: -60px;
                        }
                    
      </style>""", unsafe_allow_html=True)
      
def footer():
    st.markdown("""
     <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #E5E5E5;
            color: black;
            text-align: center;
            }
    </style>
    """, unsafe_allow_html=True)
    
      
