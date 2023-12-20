# import requests
# import os
# from template.constant import API_CHRONOLIFE_URL_ROOT
# import template.api as api

# def save_credentials(username, api_key):
#     path_creds = os.getcwd() + '/secret/' + username
#     if not os.path.exists(path_creds):
#         os.mkdir(path_creds)
        
#     with open(path_creds + '/api_ids.txt', 'w') as f:
#         f.write('user = ' + username + '\ntoken = ' + api_key + '\nurl = ' + API_CHRONOLIFE_URL_ROOT + "/data")
    
#     return path_creds
        
# def remove_credentials(username):
#     path_creds = os.getcwd() + '/secret/' + username
#     os.remove(path_creds + '/api_ids.txt')
#     os.rmdir(path_creds)