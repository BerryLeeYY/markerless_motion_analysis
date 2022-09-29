import pandas as pd
import pymysql
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import cv2
import mediapipe as mp
from numpy.linalg import norm 
import math
import plotly.graph_objs as go



app = dash.Dash(__name__, suppress_callback_exceptions=True)

conn=pymysql.connect(host='',port=int(),user='',passwd='',db='motion_analysis_db')
df=pd.read_sql_query("SELECT * FROM user_registration",conn)


ID_Password_dict_list=dict([(ID,Password) for ID, Password in zip(df.user_id, df.user_password)])
balance_df=pd.read_sql_query("SELECT * FROM user_performance_table",conn)



fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 1, 2])])
fig.update_layout(
    margin=dict(l=40, r=40, t=40, b=40)
)



######################################################################################################
######################################################################################################

def video_processing(video_data):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    draw_video_array = []
    right_tose = []
    left_tose = []
    right_knee = []
    left_knee = []
    right_pelvic = []
    left_pelvic = []
    right_hand = []
    left_hand = []
    strn = []
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        for frame in video_data:
            if frame is None:
                break
            image = frame
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            right_tose_coor = [results.pose_landmarks.landmark[32].x, results.pose_landmarks.landmark[32].y, results.pose_landmarks.landmark[32].z]
            left_tose_coor = [results.pose_landmarks.landmark[31].x, results.pose_landmarks.landmark[31].y, results.pose_landmarks.landmark[31].z]
            right_tose.append(right_tose_coor)
            left_tose.append(left_tose_coor)

            right_knee_coor = [results.pose_landmarks.landmark[26].x, results.pose_landmarks.landmark[26].y, results.pose_landmarks.landmark[26].z]
            left_knee_coor = [results.pose_landmarks.landmark[25].x, results.pose_landmarks.landmark[25].y, results.pose_landmarks.landmark[25].z]
            right_knee.append(right_knee_coor)
            left_knee.append(left_knee_coor)


            right_pelvic_coor = [results.pose_landmarks.landmark[24].x, results.pose_landmarks.landmark[24].y, results.pose_landmarks.landmark[24].z]
            left_pelvic_coor = [results.pose_landmarks.landmark[23].x, results.pose_landmarks.landmark[23].y, results.pose_landmarks.landmark[23].z]
            right_pelvic.append(right_pelvic_coor)
            left_pelvic.append(left_pelvic_coor)


            right_hand_coor = [results.pose_landmarks.landmark[20].x, results.pose_landmarks.landmark[20].y, results.pose_landmarks.landmark[20].z]
            left_hand_coor = [results.pose_landmarks.landmark[19].x, results.pose_landmarks.landmark[19].y, results.pose_landmarks.landmark[19].z]
            right_hand.append(right_hand_coor)
            left_hand.append(left_hand_coor)
            
            strn_coor = [(results.pose_landmarks.landmark[11].x + results.pose_landmarks.landmark[12].x)/2, (results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y)/2, (results.pose_landmarks.landmark[11].z + results.pose_landmarks.landmark[12].z)/2]
            strn.append(strn_coor)


            draw_video_array.append(image)
            # Flip the image horizontally for a selfie-view display.
            #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

    return [draw_video_array, right_tose, left_tose, right_knee, left_knee, right_pelvic, left_pelvic, right_hand, left_hand, strn]

######################################################################################################
######################################################################################################
def lifting_count_func(right_tose, left_tose):
    lifting_foot_count = 0
    time = len(right_tose)
    right_tose_df = pd.DataFrame(right_tose, columns = ["x", "y", "z"])
    left_tose_df = pd.DataFrame(left_tose, columns = ["x", "y", "z"])

    right_tose_height = right_tose_df["y"]
    left_tose_height = left_tose_df["y"]


    if abs(right_tose_height.mean()) > abs(left_tose_height.mean()):
        landing_foot_height = left_tose_height
        threshold = ((landing_foot_height.mean())+ 3.5*(landing_foot_height.std()))
        for i in range(time):
            try:
                if landing_foot_height[i] < threshold and landing_foot_height[i+1] > threshold and landing_foot_height[i+5] > threshold:
                    lifting_foot_count = lifting_foot_count + 1
            except:
                continue
    if abs(right_tose_height.mean()) < abs(left_tose_height.mean()):
        landing_foot_height = right_tose_height
        threshold = ((landing_foot_height.mean())+ 3.5*(landing_foot_height.std()))
        for i in range(time):
            try:
                if landing_foot_height[i] < threshold and landing_foot_height[i+1] > threshold and landing_foot_height[i+5] > threshold:
                    lifting_foot_count = lifting_foot_count + 1
            except:
                continue
    if lifting_foot_count > 12:
        lifting_foot_count = 12

    return [lifting_foot_count, landing_foot_height, threshold]

######################################################################################################
######################################################################################################
def hand_off_count_func(right_hand, left_hand):
    hand_iliac_count = 0;
    time = len(right_hand)
    right_hand_df = pd.DataFrame(right_hand, columns = ["x", "y", "z"])
    left_hand_df = pd.DataFrame(left_hand, columns = ["x", "y", "z"])

    R_L_distance = right_hand_df["x"] - left_hand_df["x"]
    hand_threshold = (R_L_distance.mean())+(R_L_distance.std())*3.5
    for i in range(time):
        try:
            if abs(R_L_distance[i]) < (hand_threshold) and  abs(R_L_distance[i+1]) > (hand_threshold): 
                hand_iliac_count = hand_iliac_count + 1
        except:
            continue
            
    if hand_iliac_count > 12:
        hand_iliac_count = 12
        
    return [hand_iliac_count, R_L_distance, hand_threshold]

######################################################################################################
######################################################################################################
def leg_angle_count_func(right_pelvic, left_pelvic, right_knee, left_knee, strn):
    R_angle_all = []
    L_angle_all = []
    time = len(right_pelvic)
    COM = (np.array(right_pelvic) + np.array(left_pelvic))/2
    V_STRN_COM = COM[:,:-1] - np.array(strn)[:,:-1]
    V_RASI_knee = np.array(right_knee)[:,:-1] - np.array(right_pelvic)[:,:-1]
    V_LASI_knee = np.array(left_knee)[:,:-1] - np.array(left_pelvic)[:,:-1]

    #cos(angle) = dot(A,B) / (norm(A).*norm(B))

    for vector_num in range(V_STRN_COM.shape[0]):
        R_cos_value = np.dot(V_STRN_COM[vector_num], V_RASI_knee[vector_num]) / (norm(V_STRN_COM[vector_num], 2)*norm(V_RASI_knee[vector_num], 2))
        R_angle_all.append(math.acos(R_cos_value)* 180 / math.pi)

        L_cos_value = np.dot(V_STRN_COM[vector_num], V_LASI_knee[vector_num]) / (norm(V_STRN_COM[vector_num], 2)*norm(V_LASI_knee[vector_num], 2))
        L_angle_all.append(math.acos(L_cos_value)* 180 / math.pi)

    R_angle_all = np.array(R_angle_all)
    L_angle_all = np.array(L_angle_all)

    angle_count = 0
    if abs(np.mean(R_angle_all)) > abs(np.mean(L_angle_all)): ### right leg main moving leg
        main_moving_leg = R_angle_all
        for i in range(time-1):
            if abs(R_angle_all[i]- abs(np.mean(R_angle_all[:150]))) < 12 and  abs(R_angle_all[i+1] - abs(np.mean(R_angle_all[:150]))) > 12:
                angle_count = angle_count + 1

    elif abs(np.mean(R_angle_all)) < abs(np.mean(L_angle_all)): ### left leg main moving leg
        main_moving_leg = L_angle_all
        for i in range(time-1):
            if abs(L_angle_all[i]- abs(np.mean(L_angle_all[:150]))) < 12 and  abs(L_angle_all[i+1]- abs(np.mean(L_angle_all[:150]))) > 12:
                angle_count = angle_count + 1
    
    if angle_count > 12:
        angle_count = 12
    
    return [angle_count, main_moving_leg]

######################################################################################################
######################################################################################################
def falling_count_func(right_tose, left_tose):
    falling_count = 0;
    time = len(right_tose)
    right_tose_df = pd.DataFrame(right_tose, columns = ["x", "y", "z"])
    left_tose_df = pd.DataFrame(left_tose, columns = ["x", "y", "z"])

    R_L_distance = right_tose_df["y"] - left_tose_df["y"]
    for i in range(time):
        try:
            if abs(R_L_distance[i]) > 0.05 and abs(R_L_distance[i+1]) < 0.05 : 
                falling_count = falling_count + 1
        except:
            continue
    
    if falling_count > 12:
        falling_count = 12
    
    return [falling_count, R_L_distance]

######################################################################################################
######################################################################################################

def remaining_out_count_func(video_array, landing_foot_height, threshold, main_moving_leg, R_L_distance, hand_threshold, R_L_distance_foot):
    remain_time = round(len(video_array)/12)
    time = len(R_L_distance_foot)
    out_position_count = 0;
    for i in range(time):
        try:
            if (sum((landing_foot_height[i:i+remain_time]) > threshold)>=remain_time) or (sum((main_moving_leg[i:i+remain_time]) > 30)>= remain_time) or (sum((R_L_distance[i:i+remain_time]) > hand_threshold) >= remain_time) or (sum(R_L_distance_foot[1:1+remain_time]< 0.05) >= remain_time):   
                out_position_count = out_position_count + 1;
        except:
            continue
    if out_position_count > 12:
        out_position_count = 12
    return out_position_count

######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################

app.layout = html.Div([
	dcc.Location(id = 'url', refresh = False),
    html.Div(id = 'page-content'),
    dcc.Store(id='current_user_id', storage_type='session')
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    )

def display_page(pathname):
    if pathname == "/main_page":
        return main_page
    else:
        return login_page

##############################################################################################

login_page = html.Div([
                        html.Div(" ", style = {'width':'700px', 'height':'1300px'}),
                        html.Div([
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Div([
                                html.Br(),
                                html.Div("Login", style = {'font-size':'40px',
                                                            'border-left': 'double',
                                                            'border-left-color':'blue',
                                                            'padding': '10px',
                                                            'width':'80%',
                                                            'margin-left': '10px'}),
                                html.Br(),
                                html.Br(),
                                html.Div('Please fill out the follow question to register your account', style = {'margin-left': '10px'}),
                                html.Br(),
                                html.Br(),
                                html.Div('Account:', style = {'margin-left': '10px'}),
                                dcc.Input(id = 'account_id', 
                                        placeholder = 'Enter your account',
                                        type = 'text',
                                        style = {'width':'250px', 'margin-left': '10px'}
                                        ),
                                html.Br(),
                                html.Br(),
                                html.Div('Password:', style = {'margin-left': '10px'}),
                                dcc.Input(id = 'password_id', 
                                        placeholder = 'Enter your password',
                                        type = 'password',
                                        style = {'width':'250px', 'margin-left': '10px'}
                                        ),
                                html.Br(),
                                html.Br(),
                                html.Button(
                                            'login',
                                            id = 'Login_button',
                                            style = {'margin-left': '10px'}
                                ),
                                html.Br(),
                                html.Br(),
                                html.Div("", id = 'register_return', style = {'margin-left': '10px'})

                            ], style = {'border-style':'groove', 'border-radius':'25px', 'height':'500px'})
                        ], style = {'width':'700px', 'height':'1300px'}),
                        html.Div(" ", style = {'width':'700px', 'height':'1300px'})
], style = {'display':'flex'})


@app.callback(
    Output('register_return', 'children'),
    Output('current_user_id', 'data'),
    Input('Login_button', 'n_clicks'),
    State('account_id', 'value'),
    State('password_id', 'value'),
)

def login_func(n_clicks, account_id, password_id):
    conn=pymysql.connect()
    df=pd.read_sql_query("SELECT * FROM user_registration",conn)
    conn.close()
    

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if ('Login_button' in  changed_id):
        try :
            if ID_Password_dict_list[account_id] == password_id:
                return_div = html.Div(dcc.Link('personal dashboard', href = "/main_page"))
                user_id = account_id
            else:
                return_div = html.Div("Wrong password!", style = {"color":"red"})
                user_id = None
        except:
            if account_id is None:
                return_div = html.Div(" ")
                user_id = None
            else:
                return_div = html.Div("Wrong account or password!", style = {"color":"red"})
                user_id = None
    else:
        return_div = html.Div(' ')
        user_id = None
    return [return_div, user_id]


############################################################################

main_page = html.Div([
                html.Div("", style = {'height':'80px', "background-color":"#A9A9A9"}),
                html.Div("", style = {'height':'30px', "background-color":"#DCDCDC"}),
                html.Div([
                    html.Div("", style = {'width':'100px', 'height':'1000px'}),
                    html.Div([
                        html.Div("", 
                                id = "name_id",
                                style = {'font-size':'40px','margin-left': '10px', "font-family":'Arial Black', "padding":"10px 0"}),
                        html.Br(),
                        html.Br(),
                        html.Button("Download history", id="download_button_id", style = { 'width':'150px', 'height':'60px', 'font-size':'20px'}),
                        dcc.Download(id="download_id")
                        ], style = {'width':'250px', 'height':'1000px'}),

                    html.Div([], style = {'width':'15px', 'height':'1000px'}),
                    html.Div([
                        html.Img(src = "/assets/balance_performance_index.png", style = {'width':'500px', 'height':'150px'}),
                        html.Br(),
                        html.Br(),
                        html.Div([
                            html.Div("Your current balance performance", style = {'font-size':'30px','margin-left': '10px', "font-family":'Arial Black', "padding":"5px 0"}),
                            html.Br(),
                            html.Div('', id = 'balance_ability_id')
                            ], style = {'height':'490px','border-style':'groove', 'border-radius':'15px', "background-color":"white"})
                        
                    ],
                    id = 'first_part_dashboard', 
                    style = {'margin-left': '10px', 'width':'500px', 'height':'1000px'}),
                    html.Div([], style = {'width':'50px', 'height':'1000px'}),
                    html.Div([
                        html.Div([
                            html.Div("0", id = "circle_1_id", style = {'width':'150px', 'border-style':'groove', 'border-radius':'15px', "background-color":"white", "text-align": "center", "padding":"15px 0","font-size":"30px", "font-family":'Arial Black'}),
                            html.Div("", style = {'width':'1px'}),
                            html.Div("0", id = "circle_2_id", style = {'width':'150px', 'border-style':'groove', 'border-radius':'15px', "background-color":"white", "text-align": "center", "padding":"15px 0","font-size":"30px", "font-family":"Arial Black"}),
                            html.Div("", style = {'width':'1px'}),
                            html.Div("0", id = "circle_3_id", style = {'width':'150px', 'border-style':'groove', 'border-radius':'15px', "background-color":"white", "text-align": "center", "padding":"15px 0","font-size":"30px", "font-family":"Arial Black"}),
                            html.Div("", style = {'width':'1px'}),
                            html.Div("0", id = "circle_4_id", style = {'width':'150px', 'border-style':'groove', 'border-radius':'15px', "background-color":"white", "text-align": "center", "padding":"15px 0","font-size":"30px", "font-family":"Arial Black"}),
                            html.Div("", style = {'width':'1px'}),
                            html.Div("0", id = "circle_5_id", style = {'width':'150px', 'border-style':'groove', 'border-radius':'15px', "background-color":"white", "text-align": "center", "padding":"15px 0","font-size":"30px", "font-family":"Arial Black"}),
                        ], style = {'width':'750px', 'height':'150px', 'display':'flex'}),
                        html.Div([ ], style = {'width':'750px', 'height':'10px'}),
                        html.Div([
                            html.Br(),
                            dcc.Graph(figure=fig, 
                                        id = "graph_id",
                                        style = {'margin-left': '5px'}
                            )], id = "flow_chart_id", style = {'width':'750px', 'height':'500px', 'border-style':'groove', 'border-radius':'15px', "background-color":"white"})
                    ], style = {'width':'750px', 'height':'1000px'}),
                    html.Div([], style = {'width':'20px', 'height':'1000px'}),
                ], style = {"display":"flex", "background-color":"#DCDCDC"})
            ])

@app.callback(
    Output('name_id', 'children'),
    Output('balance_ability_id', 'children'),
    Output('graph_id', 'figure'),
    Input('current_user_id', 'data')

)


def display_balance_func(current_user_id):
    name = current_user_id
    user_balance_df = balance_df[balance_df["user_id"]==name]
    user_balance_latest = user_balance_df.iloc[-1,6]
    y = user_balance_df.performance_all
    x = user_balance_df.performing_date
    fig = go.Figure(data=[go.Scatter(x=x, y=y)])
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        title = "Balance performance vs Time",
        title_x=0.5,
        xaxis_title="Date", 
        yaxis_title="Balance error sum"
    )



    name_div = html.Div("Hello " + name)
    if user_balance_latest < 5:
        balance_return_div = html.Div([
                                        html.Div(str(user_balance_df.iloc[-1,6]), style = {'width':'475px','margin-left': '10px', 'border-radius':'25px', "background-color":"green", "text-align": "center", "padding":"45px 0","font-size":"30px", "font-family":"Arial Black"}),
                                        html.Br(),
                                        html.Div(["Foot lifting: " + str(user_balance_df.iloc[-1,1])], style = {'margin-left': '10px', "font-size":"30px","font-family":'Arial Black' }),
                                        html.Div(["Hand off: " + str(user_balance_df.iloc[-1,2])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        html.Div(["Hip motion: " + str(user_balance_df.iloc[-1,3])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        html.Div(["Stepping: " + str(user_balance_df.iloc[-1,4])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        html.Div(["Out position: " + str(user_balance_df.iloc[-1,5])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        ])
    elif (user_balance_latest >= 5) and (user_balance_latest < 12):
        balance_return_div = html.Div([
                                        html.Div(str(user_balance_df.iloc[-1,6]), style = {'width':'475px','margin-left': '10px', 'border-radius':'25px', "background-color":"yellow", "text-align": "center", "padding":"45px 0","font-size":"30px", "font-family":"Arial Black"}),
                                        html.Br(),
                                        html.Div(["Foot lifting: " + str(user_balance_df.iloc[-1,1])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        html.Div(["Hand off: " + str(user_balance_df.iloc[-1,2])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        html.Div(["Hip motion: " + str(user_balance_df.iloc[-1,3])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        html.Div(["Stepping: " + str(user_balance_df.iloc[-1,4])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        html.Div(["Out position: " + str(user_balance_df.iloc[-1,5])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        ])
    elif user_balance_latest >= 12:
        balance_return_div = html.Div([
                                        html.Div(str(user_balance_df.iloc[-1,6]), style = {'width':'475px','margin-left': '10px', 'border-radius':'25px', "background-color":"red", "text-align": "center", "padding":"45px 0","font-size":"30px", "font-family":"Arial Black"}),
                                        html.Br(),
                                        html.Div(["Foot lifting: " + str(user_balance_df.iloc[-1,1])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        html.Div(["Hand off: " + str(user_balance_df.iloc[-1,2])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        html.Div(["Hip motion: " + str(user_balance_df.iloc[-1,3])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        html.Div(["Stepping: " + str(user_balance_df.iloc[-1,4])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        html.Div(["Out position: " + str(user_balance_df.iloc[-1,5])], style = {'margin-left': '10px', "font-size":"30px", "font-family":'Arial Black' }),
                                        ])
            

    return [name_div, balance_return_div, fig]




@app.callback(
    Output('circle_1_id', 'children'),
    Output('circle_2_id', 'children'),
    Output('circle_3_id', 'children'),
    Output('circle_4_id', 'children'),
    Output('circle_5_id', 'children'),
    Input('current_user_id', 'data'),
)

def video_analysis_func(user_id):

    user_df = balance_df[balance_df["user_id"]==user_id]
    lifting_foot_count = user_df["performance_1"].mean()
    hand_iliac_count = user_df["performance_2"].mean()
    angle_count = user_df["performance_3"].mean()
    falling_count = user_df["performance_4"].mean()
    out_position_count = user_df["performance_5"].mean()

    circle_1_div = html.Div([
                            html.Div("Average foot_lifting", style = {"font-size":"20px"}),
                            html.Div(round(lifting_foot_count)),
                            ])
    circle_2_div = html.Div([
                            html.Div("Average hand_off", style = {"font-size":"20px"}),
                            html.Div(round(hand_iliac_count)),
                            ])
    circle_3_div = html.Div([
                            html.Div("Average hip_motion", style = {"font-size":"20px"}),
                            html.Div(round(angle_count)),
                            ])
    circle_4_div = html.Div([
                            html.Div("Average stepping", style = {"font-size":"20px"}),
                            html.Div(round(falling_count)),
                            ])
    circle_5_div = html.Div([
                            html.Div("Average out_position", style = {"font-size":"20px"}),
                            html.Div(round(out_position_count)),
                            ])
    
    return [circle_1_div, circle_2_div, circle_3_div, circle_4_div, circle_5_div]


@app.callback(
    Output("download_id", "data"),
    Input('current_user_id', 'data'),
    Input("download_button_id", "n_clicks"),
    prevent_initial_call=True,
)

def download_func(name, n_clicks):
    user_balance_df = balance_df[balance_df["user_id"]==name]
    return dcc.send_data_frame(user_balance_df.to_csv, "balance_performance.csv")

#########################################################################3







if __name__ == '__main__':
	app.run_server(debug=True)
