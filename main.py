# import os
# from datetime import datetime
# import json
# import csv
# from io import StringIO
# import pandas as pd
# from flask import Flask, render_template, request, jsonify, send_file
# from werkzeug.utils import secure_filename
# from google.generativeai import GenerativeModel, configure
# import requests
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
# app.secret_key = os.getenv('FLASK_SECRET_KEY')

# # Configure AI services
# configure(api_key=os.getenv('GEMINI_API_KEY'))
# gemini_model = GenerativeModel('gemini-2.5-flash')
# HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# # In-memory database (replace with real DB in production)
# users_db = {}
# workouts_db = {}
# goals_db = {}

# # Ensure upload directory exists
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# class FitnessTracker:
#     def __init__(self):
#         self.users = users_db
#         self.workouts = workouts_db
#         self.goals = goals_db

#     def add_user(self, user_id, name, age, weight, height, fitness_level):
#         self.users[user_id] = {
#             'name': name,
#             'age': age,
#             'weight': weight,
#             'height': height,
#             'fitness_level': fitness_level,
#             'created_at': datetime.now().isoformat()
#         }
#         return True

#     def log_workout(self, user_id, workout_type, duration, calories, intensity, notes=""):
#         if user_id not in self.workouts:
#             self.workouts[user_id] = []
        
#         workout = {
#             'id': len(self.workouts[user_id]) + 1,
#             'type': workout_type,
#             'duration': duration,  # in minutes
#             'calories': calories,
#             'intensity': intensity,
#             'notes': notes,
#             'timestamp': datetime.now().isoformat()
#         }
#         self.workouts[user_id].append(workout)
#         return workout

#     def set_goal(self, user_id, goal_type, target_value, target_date):
#         if user_id not in self.goals:
#             self.goals[user_id] = []
        
#         goal = {
#             'id': len(self.goals[user_id]) + 1,
#             'type': goal_type,
#             'target_value': target_value,
#             'current_value': 0,
#             'target_date': target_date,
#             'created_at': datetime.now().isoformat(),
#             'completed': False
#         }
#         self.goals[user_id].append(goal)
#         return goal

#     def get_user_stats(self, user_id):
#         if user_id not in self.users:
#             return None
        
#         stats = {
#             'user_info': self.users[user_id],
#             'total_workouts': len(self.workouts.get(user_id, [])),
#             'total_calories': sum(w['calories'] for w in self.workouts.get(user_id, [])),
#             'active_goals': len([g for g in self.goals.get(user_id, []) if not g['completed']]),
#             'completed_goals': len([g for g in self.goals.get(user_id, []) if g['completed']]),
#             'workouts': self.workouts.get(user_id, []),  # Add actual workout data
#             'goals': self.goals.get(user_id, [])        # Add actual goal data
#         }
#         return stats
#         def get_workout_recommendation(self, user_id):
#             if user_id not in self.users:
#                 return None
            
#             user = self.users[user_id]
#             prompt = f"""
#             Based on the following user profile, suggest 3 personalized workout routines:
#             - Name: {user['name']}
#             - Age: {user['age']}
#             - Weight: {user['weight']} kg
#             - Height: {user['height']} cm
#             - Fitness Level: {user['fitness_level']}
            
#             The user has completed {len(self.workouts.get(user_id, []))} workouts so far.
#             Provide recommendations that consider their fitness level and would help them progress.
#             Format the response as a markdown list with brief explanations for each recommendation.
#             """
            
#             try:
#                 response = gemini_model.generate_content(prompt)
#                 return response.text
#             except Exception as e:
#                 print(f"Error generating recommendation: {e}")
#                 return None

#         def analyze_workout_sentiment(self, workout_notes):
#             if not workout_notes:
#                 return None
            
#             API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
#             headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
            
#             try:
#                 response = requests.post(API_URL, headers=headers, json={"inputs": workout_notes})
#                 result = response.json()
                
#                 if isinstance(result, list) and len(result) > 0:
#                     sentiment = result[0][0]['label']
#                     return sentiment
#                 return None
#             except Exception as e:
#                 print(f"Error analyzing sentiment: {e}")
#                 return None

#         def export_data(self, user_id, format='json'):
#             if user_id not in self.users:
#                 return None
            
#             data = {
#                 'user': self.users[user_id],
#                 'workouts': self.workouts.get(user_id, []),
#                 'goals': self.goals.get(user_id, [])
#             }
            
#             if format == 'json':
#                 return json.dumps(data, indent=2)
#             elif format == 'csv':
#                 output = StringIO()
#                 writer = csv.writer(output)
                
#                 # Write user info
#                 writer.writerow(['User Information'])
#                 writer.writerow(['Name', 'Age', 'Weight', 'Height', 'Fitness Level'])
#                 writer.writerow([
#                     data['user']['name'],
#                     data['user']['age'],
#                     data['user']['weight'],
#                     data['user']['height'],
#                     data['user']['fitness_level']
#                 ])
                
#                 # Write workouts
#                 writer.writerow([])
#                 writer.writerow(['Workouts'])
#                 writer.writerow(['ID', 'Type', 'Duration', 'Calories', 'Intensity', 'Notes', 'Timestamp'])
#                 for workout in data['workouts']:
#                     writer.writerow([
#                         workout['id'],
#                         workout['type'],
#                         workout['duration'],
#                         workout['calories'],
#                         workout['intensity'],
#                         workout['notes'],
#                         workout['timestamp']
#                     ])
                
#                 # Write goals
#                 writer.writerow([])
#                 writer.writerow(['Goals'])
#                 writer.writerow(['ID', 'Type', 'Target Value', 'Current Value', 'Target Date', 'Completed'])
#                 for goal in data['goals']:
#                     writer.writerow([
#                         goal['id'],
#                         goal['type'],
#                         goal['target_value'],
#                         goal['current_value'],
#                         goal['target_date'],
#                         goal['completed']
#                     ])
                
#                 return output.getvalue()
#             elif format == 'excel':
#                 df_user = pd.DataFrame([data['user']])
#                 df_workouts = pd.DataFrame(data['workouts'])
#                 df_goals = pd.DataFrame(data['goals'])
                
#                 output = StringIO()
#                 with pd.ExcelWriter(output, engine='openpyxl') as writer:
#                     df_user.to_excel(writer, sheet_name='User Info', index=False)
#                     df_workouts.to_excel(writer, sheet_name='Workouts', index=False)
#                     df_goals.to_excel(writer, sheet_name='Goals', index=False)
                
#                 return output.getvalue()
#             else:
#                 return None

# # Initialize tracker
# tracker = FitnessTracker()

# # Routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/users', methods=['POST'])
# def add_user():
#     data = request.json
#     required_fields = ['user_id', 'name', 'age', 'weight', 'height', 'fitness_level']
    
#     if not all(field in data for field in required_fields):
#         return jsonify({'error': 'Missing required fields'}), 400
    
#     tracker.add_user(
#         data['user_id'],
#         data['name'],
#         data['age'],
#         data['weight'],
#         data['height'],
#         data['fitness_level']
#     )
#     return jsonify({'message': 'User added successfully'}), 201

# @app.route('/api/users/<user_id>/workouts', methods=['POST'])
# def log_workout(user_id):
#     data = request.json
#     required_fields = ['workout_type', 'duration', 'calories', 'intensity']
    
#     if not all(field in data for field in required_fields):
#         return jsonify({'error': 'Missing required fields'}), 400
    
#     notes = data.get('notes', '')
#     workout = tracker.log_workout(
#         user_id,
#         data['workout_type'],
#         data['duration'],
#         data['calories'],
#         data['intensity'],
#         notes
#     )
    
#     # Analyze sentiment if notes are provided
#     sentiment = None
#     if notes:
#         sentiment = tracker.analyze_workout_sentiment(notes)
    
#     response = {
#         'message': 'Workout logged successfully',
#         'workout': workout,
#         'sentiment': sentiment
#     }
#     return jsonify(response), 201

# @app.route('/api/users/<user_id>/goals', methods=['POST'])
# def set_goal(user_id):
#     data = request.json
#     required_fields = ['goal_type', 'target_value', 'target_date']
    
#     if not all(field in data for field in required_fields):
#         return jsonify({'error': 'Missing required fields'}), 400
    
#     goal = tracker.set_goal(
#         user_id,
#         data['goal_type'],
#         data['target_value'],
#         data['target_date']
#     )
#     return jsonify({'message': 'Goal set successfully', 'goal': goal}), 201
# @app.route('/api/users/<user_id>/stats', methods=['GET'])
# def get_stats(user_id):
#     stats = tracker.get_user_stats(user_id)
#     if not stats:
#         return jsonify({'error': 'User not found'}), 404
    
#     # Add workouts and goals to the response
#     stats['workouts'] = tracker.workouts.get(user_id, [])
#     stats['goals'] = tracker.goals.get(user_id, [])
    
#     return jsonify(stats)
# # @app.route('/api/users/<user_id>/stats', methods=['GET'])
# # def get_stats(user_id):
# #     stats = tracker.get_user_stats(user_id)
# #     if not stats:
# #         return jsonify({'error': 'User not found'}), 404
# #     return jsonify(stats)

# @app.route('/api/users/<user_id>/recommendations', methods=['GET'])
# def get_recommendations(user_id):
#     recommendation = tracker.get_workout_recommendation(user_id)
#     if not recommendation:
#         return jsonify({'error': 'Unable to generate recommendations'}), 500
#     return jsonify({'recommendation': recommendation})

# @app.route('/api/users/<user_id>/export', methods=['GET'])
# def export_data(user_id):
#     format = request.args.get('format', 'json')
    
#     if format not in ['json', 'csv', 'excel']:
#         return jsonify({'error': 'Invalid format specified'}), 400
    
#     data = tracker.export_data(user_id, format)
#     if not data:
#         return jsonify({'error': 'User not found'}), 404
    
#     filename = f"fitness_data_{user_id}_{datetime.now().strftime('%Y%m%d')}"
    
#     if format == 'json':
#         filename += '.json'
#         return send_file(
#             StringIO(data),
#             mimetype='application/json',
#             as_attachment=True,
#             download_name=filename
#         )
#     elif format == 'csv':
#         filename += '.csv'
#         return send_file(
#             StringIO(data),
#             mimetype='text/csv',
#             as_attachment=True,
#             download_name=filename
#         )
#     elif format == 'excel':
#         filename += '.xlsx'
#         return send_file(
#             StringIO(data),
#             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
#             as_attachment=True,
#             download_name=filename
#         )

# @app.route('/api/ai/generate', methods=['POST'])
# def generate_content():
#     data = request.json
#     prompt = data.get('prompt')
    
#     if not prompt:
#         return jsonify({'error': 'Prompt is required'}), 400
    
#     try:
#         response = gemini_model.generate_content(prompt)
#         return jsonify({'response': response.text})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # File upload and processing
# ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx'}

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/api/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         try:
#             # Process the file based on its type
#             if filename.endswith('.csv'):
#                 df = pd.read_csv(filepath)
#             elif filename.endswith('.json'):
#                 df = pd.read_json(filepath)
#             elif filename.endswith('.xlsx'):
#                 df = pd.read_excel(filepath)
#             else:
#                 return jsonify({'error': 'Unsupported file format'}), 400
            
#             # Generate analysis using AI
#             prompt = f"""
#             Analyze this fitness data and provide insights:
#             {df.head().to_string()}
            
#             Provide:
#             1. Key statistics
#             2. Notable trends
#             3. Recommendations for improvement
#             """
            
#             analysis = gemini_model.generate_content(prompt).text
            
#             return jsonify({
#                 'message': 'File processed successfully',
#                 'analysis': analysis,
#                 'data_preview': df.head().to_dict()
#             })
#         except Exception as e:
#             return jsonify({'error': f'Error processing file: {str(e)}'}), 500
#     else:
#         return jsonify({'error': 'Invalid file type'}), 400

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)












# # import os
# # from datetime import datetime
# # import json
# # import csv
# # from io import StringIO
# # import pandas as pd
# # from flask import Flask, render_template, request, jsonify, send_file
# # from werkzeug.utils import secure_filename
# # from google.generativeai import GenerativeModel, configure
# # import requests
# # from dotenv import load_dotenv
# # import mysql.connector
# # from mysql.connector import Error
# # from flask_cors import CORS

# # # Load environment variables
# # load_dotenv()

# # app = Flask(__name__)
# # CORS(app)
# # app.config['UPLOAD_FOLDER'] = 'uploads'
# # app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
# # app.secret_key = os.getenv('FLASK_SECRET_KEY')

# # # Configure AI services
# # configure(api_key=os.getenv('GEMINI_API_KEY'))
# # gemini_model = GenerativeModel('gemini-2.5-flash')
# # HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# # # Database configuration
# # DB_CONFIG = {
# #     'host': os.getenv('DB_HOST', 'localhost'),
# #     'database': os.getenv('DB_NAME', 'fitness_tracker'),
# #     'user': os.getenv('DB_USER', 'root'),
# #     'password': os.getenv('DB_PASSWORD', 'sriparnakoar'),  # Make sure this matches your MySQL root password
# #     'port': int(os.getenv('DB_PORT', 3306))  # Convert port to integer
# # }

# # # Ensure upload directory exists
# # os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # def create_db_connection():
# #     """Create and return a database connection"""
# #     try:
# #         connection = mysql.connector.connect(**DB_CONFIG)
# #         return connection
# #     except Error as e:
# #         print(f"Error connecting to MySQL: {e}")
# #         return None

# # def initialize_database():
# #     """Initialize the database with required tables"""
# #     connection = create_db_connection()
# #     if connection is None:
# #         return False
    
# #     try:
# #         cursor = connection.cursor()
        
# #         # Create users table
# #         cursor.execute("""
# #         CREATE TABLE IF NOT EXISTS users (
# #             user_id VARCHAR(36) PRIMARY KEY,
# #             name VARCHAR(100) NOT NULL,
# #             age INT NOT NULL,
# #             weight FLOAT NOT NULL,
# #             height FLOAT NOT NULL,
# #             fitness_level VARCHAR(50) NOT NULL,
# #             created_at DATETIME NOT NULL
# #         )
# #         """)
        
# #         # Create workouts table
# #         cursor.execute("""
# #         CREATE TABLE IF NOT EXISTS workouts (
# #             id INT AUTO_INCREMENT PRIMARY KEY,
# #             user_id VARCHAR(36) NOT NULL,
# #             workout_type VARCHAR(50) NOT NULL,
# #             duration INT NOT NULL,
# #             calories INT NOT NULL,
# #             intensity VARCHAR(20) NOT NULL,
# #             notes TEXT,
# #             timestamp DATETIME NOT NULL,
# #             FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
# #         )
# #         """)
        
# #         # Create goals table
# #         cursor.execute("""
# #         CREATE TABLE IF NOT EXISTS goals (
# #             id INT AUTO_INCREMENT PRIMARY KEY,
# #             user_id VARCHAR(36) NOT NULL,
# #             goal_type VARCHAR(50) NOT NULL,
# #             target_value FLOAT NOT NULL,
# #             current_value FLOAT DEFAULT 0,
# #             target_date DATE NOT NULL,
# #             created_at DATETIME NOT NULL,
# #             completed BOOLEAN DEFAULT FALSE,
# #             FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
# #         )
# #         """)
        
# #         connection.commit()
# #         return True
# #     except Error as e:
# #         print(f"Error initializing database: {e}")
# #         return False
# #     finally:
# #         if connection.is_connected():
# #             cursor.close()
# #             connection.close()

# # class FitnessTracker:
# #     def __init__(self):
# #         initialize_database()

# #     def add_user(self, user_id, name, age, weight, height, fitness_level):
# #         connection = create_db_connection()
# #         if connection is None:
# #             return False
        
# #         try:
# #             cursor = connection.cursor()
# #             query = """
# #             INSERT INTO users (user_id, name, age, weight, height, fitness_level, created_at)
# #             VALUES (%s, %s, %s, %s, %s, %s, %s)
# #             """
# #             cursor.execute(query, (user_id, name, age, weight, height, fitness_level, datetime.now()))
# #             connection.commit()
# #             return True
# #         except Error as e:
# #             print(f"Error adding user: {e}")
# #             return False
# #         finally:
# #             if connection.is_connected():
# #                 cursor.close()
# #                 connection.close()

# #     def get_user(self, user_id):
# #         connection = create_db_connection()
# #         if connection is None:
# #             return None
        
# #         try:
# #             cursor = connection.cursor(dictionary=True)
# #             query = "SELECT * FROM users WHERE user_id = %s"
# #             cursor.execute(query, (user_id,))
# #             return cursor.fetchone()
# #         except Error as e:
# #             print(f"Error getting user: {e}")
# #             return None
# #         finally:
# #             if connection.is_connected():
# #                 cursor.close()
# #                 connection.close()

# #     def log_workout(self, user_id, workout_type, duration, calories, intensity, notes=""):
# #         connection = create_db_connection()
# #         if connection is None:
# #             return None
        
# #         try:
# #             cursor = connection.cursor(dictionary=True)
# #             query = """
# #             INSERT INTO workouts (user_id, workout_type, duration, calories, intensity, notes, timestamp)
# #             VALUES (%s, %s, %s, %s, %s, %s, %s)
# #             """
# #             cursor.execute(query, (user_id, workout_type, duration, calories, intensity, notes, datetime.now()))
# #             connection.commit()
            
# #             # Get the inserted workout
# #             workout_id = cursor.lastrowid
# #             query = "SELECT * FROM workouts WHERE id = %s"
# #             cursor.execute(query, (workout_id,))
# #             workout = cursor.fetchone()
            
# #             return workout
# #         except Error as e:
# #             print(f"Error logging workout: {e}")
# #             return None
# #         finally:
# #             if connection.is_connected():
# #                 cursor.close()
# #                 connection.close()

# #     def set_goal(self, user_id, goal_type, target_value, target_date):
# #         connection = create_db_connection()
# #         if connection is None:
# #             return None
        
# #         try:
# #             cursor = connection.cursor(dictionary=True)
# #             query = """
# #             INSERT INTO goals (user_id, goal_type, target_value, target_date, created_at)
# #             VALUES (%s, %s, %s, %s, %s)
# #             """
# #             cursor.execute(query, (user_id, goal_type, target_value, target_date, datetime.now()))
# #             connection.commit()
            
# #             # Get the inserted goal
# #             goal_id = cursor.lastrowid
# #             query = "SELECT * FROM goals WHERE id = %s"
# #             cursor.execute(query, (goal_id,))
# #             goal = cursor.fetchone()
            
# #             return goal
# #         except Error as e:
# #             print(f"Error setting goal: {e}")
# #             return None
# #         finally:
# #             if connection.is_connected():
# #                 cursor.close()
# #                 connection.close()

# #     def get_user_stats(self, user_id):
# #         user = self.get_user(user_id)
# #         if not user:
# #             return None
        
# #         connection = create_db_connection()
# #         if connection is None:
# #             return None
        
# #         try:
# #             cursor = connection.cursor(dictionary=True)
            
# #             # Get workout stats
# #             query = """
# #             SELECT COUNT(*) as total_workouts, SUM(calories) as total_calories
# #             FROM workouts
# #             WHERE user_id = %s
# #             """
# #             cursor.execute(query, (user_id,))
# #             workout_stats = cursor.fetchone()
            
# #             # Get goal stats
# #             query = """
# #             SELECT 
# #                 SUM(CASE WHEN completed = FALSE THEN 1 ELSE 0 END) as active_goals,
# #                 SUM(CASE WHEN completed = TRUE THEN 1 ELSE 0 END) as completed_goals
# #             FROM goals
# #             WHERE user_id = %s
# #             """
# #             cursor.execute(query, (user_id,))
# #             goal_stats = cursor.fetchone()
            
# #             stats = {
# #                 'user_info': user,
# #                 'total_workouts': workout_stats['total_workouts'] if workout_stats['total_workouts'] else 0,
# #                 'total_calories': workout_stats['total_calories'] if workout_stats['total_calories'] else 0,
# #                 'active_goals': goal_stats['active_goals'] if goal_stats['active_goals'] else 0,
# #                 'completed_goals': goal_stats['completed_goals'] if goal_stats['completed_goals'] else 0
# #             }
# #             return stats
# #         except Error as e:
# #             print(f"Error getting user stats: {e}")
# #             return None
# #         finally:
# #             if connection.is_connected():
# #                 cursor.close()
# #                 connection.close()

# #     def get_user_workouts(self, user_id):
# #         connection = create_db_connection()
# #         if connection is None:
# #             return None
        
# #         try:
# #             cursor = connection.cursor(dictionary=True)
# #             query = """
# #             SELECT * FROM workouts
# #             WHERE user_id = %s
# #             ORDER BY timestamp DESC
# #             """
# #             cursor.execute(query, (user_id,))
# #             return cursor.fetchall()
# #         except Error as e:
# #             print(f"Error getting workouts: {e}")
# #             return None
# #         finally:
# #             if connection.is_connected():
# #                 cursor.close()
# #                 connection.close()

# #     def get_user_goals(self, user_id):
# #         connection = create_db_connection()
# #         if connection is None:
# #             return None
        
# #         try:
# #             cursor = connection.cursor(dictionary=True)
# #             query = """
# #             SELECT * FROM goals
# #             WHERE user_id = %s
# #             ORDER BY created_at DESC
# #             """
# #             cursor.execute(query, (user_id,))
# #             return cursor.fetchall()
# #         except Error as e:
# #             print(f"Error getting goals: {e}")
# #             return None
# #         finally:
# #             if connection.is_connected():
# #                 cursor.close()
# #                 connection.close()

# #     def get_workout_recommendation(self, user_id):
# #         user = self.get_user(user_id)
# #         if not user:
# #             return None
        
# #         workouts = self.get_user_workouts(user_id)
# #         prompt = f"""
# #         Based on the following user profile, suggest 3 personalized workout routines:
# #         - Name: {user['name']}
# #         - Age: {user['age']}
# #         - Weight: {user['weight']} kg
# #         - Height: {user['height']} cm
# #         - Fitness Level: {user['fitness_level']}
        
# #         The user has completed {len(workouts) if workouts else 0} workouts so far.
# #         Provide recommendations that consider their fitness level and would help them progress.
# #         Format the response as a markdown list with brief explanations for each recommendation.
# #         """
        
# #         try:
# #             response = gemini_model.generate_content(prompt)
# #             return response.text
# #         except Exception as e:
# #             print(f"Error generating recommendation: {e}")
# #             return None

# #     def analyze_workout_sentiment(self, workout_notes):
# #         if not workout_notes:
# #             return None
        
# #         API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
# #         headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        
# #         try:
# #             response = requests.post(API_URL, headers=headers, json={"inputs": workout_notes})
# #             result = response.json()
            
# #             if isinstance(result, list) and len(result) > 0:
# #                 sentiment = result[0][0]['label']
# #                 return sentiment
# #             return None
# #         except Exception as e:
# #             print(f"Error analyzing sentiment: {e}")
# #             return None

# #     def export_data(self, user_id, format='json'):
# #         user = self.get_user(user_id)
# #         if not user:
# #             return None
        
# #         workouts = self.get_user_workouts(user_id)
# #         goals = self.get_user_goals(user_id)
        
# #         data = {
# #             'user': user,
# #             'workouts': workouts if workouts else [],
# #             'goals': goals if goals else []
# #         }
        
# #         if format == 'json':
# #             return json.dumps(data, indent=2, default=str)
# #         elif format == 'csv':
# #             output = StringIO()
# #             writer = csv.writer(output)
            
# #             # Write user info
# #             writer.writerow(['User Information'])
# #             writer.writerow(['Name', 'Age', 'Weight', 'Height', 'Fitness Level'])
# #             writer.writerow([
# #                 data['user']['name'],
# #                 data['user']['age'],
# #                 data['user']['weight'],
# #                 data['user']['height'],
# #                 data['user']['fitness_level']
# #             ])
            
# #             # Write workouts
# #             writer.writerow([])
# #             writer.writerow(['Workouts'])
# #             writer.writerow(['ID', 'Type', 'Duration', 'Calories', 'Intensity', 'Notes', 'Timestamp'])
# #             for workout in data['workouts']:
# #                 writer.writerow([
# #                     workout['id'],
# #                     workout['workout_type'],
# #                     workout['duration'],
# #                     workout['calories'],
# #                     workout['intensity'],
# #                     workout['notes'],
# #                     workout['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(workout['timestamp'], datetime) else workout['timestamp']
# #                 ])
            
# #             # Write goals
# #             writer.writerow([])
# #             writer.writerow(['Goals'])
# #             writer.writerow(['ID', 'Type', 'Target Value', 'Current Value', 'Target Date', 'Completed'])
# #             for goal in data['goals']:
# #                 writer.writerow([
# #                     goal['id'],
# #                     goal['goal_type'],
# #                     goal['target_value'],
# #                     goal['current_value'],
# #                     goal['target_date'].strftime('%Y-%m-%d') if isinstance(goal['target_date'], datetime.date) else goal['target_date'],
# #                     'Yes' if goal['completed'] else 'No'
# #                 ])
            
# #             return output.getvalue()
# #         elif format == 'excel':
# #             df_user = pd.DataFrame([data['user']])
# #             df_workouts = pd.DataFrame(data['workouts'])
# #             df_goals = pd.DataFrame(data['goals'])
            
# #             output = StringIO()
# #             with pd.ExcelWriter(output, engine='openpyxl') as writer:
# #                 df_user.to_excel(writer, sheet_name='User Info', index=False)
# #                 df_workouts.to_excel(writer, sheet_name='Workouts', index=False)
# #                 df_goals.to_excel(writer, sheet_name='Goals', index=False)
            
# #             return output.getvalue()
# #         else:
# #             return None

# # # Initialize tracker
# # tracker = FitnessTracker()

# # # Routes
# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/api/users', methods=['POST'])
# # def add_user():
# #     data = request.json
# #     required_fields = ['user_id', 'name', 'age', 'weight', 'height', 'fitness_level']
    
# #     if not all(field in data for field in required_fields):
# #         return jsonify({'error': 'Missing required fields'}), 400
    
# #     success = tracker.add_user(
# #         data['user_id'],
# #         data['name'],
# #         data['age'],
# #         data['weight'],
# #         data['height'],
# #         data['fitness_level']
# #     )
    
# #     if not success:
# #         return jsonify({'error': 'Failed to add user'}), 500
    
# #     return jsonify({'message': 'User added successfully'}), 201

# # @app.route('/api/users/<user_id>/workouts', methods=['POST'])
# # def log_workout(user_id):
# #     data = request.json
# #     required_fields = ['workout_type', 'duration', 'calories', 'intensity']
    
# #     if not all(field in data for field in required_fields):
# #         return jsonify({'error': 'Missing required fields'}), 400
    
# #     notes = data.get('notes', '')
# #     workout = tracker.log_workout(
# #         user_id,
# #         data['workout_type'],
# #         data['duration'],
# #         data['calories'],
# #         data['intensity'],
# #         notes
# #     )
    
# #     if not workout:
# #         return jsonify({'error': 'Failed to log workout'}), 500
    
# #     # Analyze sentiment if notes are provided
# #     sentiment = None
# #     if notes:
# #         sentiment = tracker.analyze_workout_sentiment(notes)
    
# #     response = {
# #         'message': 'Workout logged successfully',
# #         'workout': workout,
# #         'sentiment': sentiment
# #     }
# #     return jsonify(response), 201

# # @app.route('/api/users/<user_id>/goals', methods=['POST'])
# # def set_goal(user_id):
# #     data = request.json
# #     required_fields = ['goal_type', 'target_value', 'target_date']
    
# #     if not all(field in data for field in required_fields):
# #         return jsonify({'error': 'Missing required fields'}), 400
    
# #     goal = tracker.set_goal(
# #         user_id,
# #         data['goal_type'],
# #         data['target_value'],
# #         data['target_date']
# #     )
    
# #     if not goal:
# #         return jsonify({'error': 'Failed to set goal'}), 500
    
# #     return jsonify({'message': 'Goal set successfully', 'goal': goal}), 201

# # @app.route('/api/users/<user_id>/stats', methods=['GET'])
# # def get_stats(user_id):
# #     stats = tracker.get_user_stats(user_id)
# #     if not stats:
# #         return jsonify({'error': 'User not found'}), 404
# #     return jsonify(stats)

# # @app.route('/api/users/<user_id>/workouts', methods=['GET'])
# # def get_workouts(user_id):
# #     workouts = tracker.get_user_workouts(user_id)
# #     if workouts is None:
# #         return jsonify({'error': 'Failed to fetch workouts'}), 500
# #     return jsonify({'workouts': workouts})

# # @app.route('/api/users/<user_id>/goals', methods=['GET'])
# # def get_goals(user_id):
# #     goals = tracker.get_user_goals(user_id)
# #     if goals is None:
# #         return jsonify({'error': 'Failed to fetch goals'}), 500
# #     return jsonify({'goals': goals})

# # @app.route('/api/users/<user_id>/recommendations', methods=['GET'])
# # def get_recommendations(user_id):
# #     recommendation = tracker.get_workout_recommendation(user_id)
# #     if not recommendation:
# #         return jsonify({'error': 'Unable to generate recommendations'}), 500
# #     return jsonify({'recommendation': recommendation})

# # @app.route('/api/users/<user_id>/export', methods=['GET'])
# # def export_data(user_id):
# #     format = request.args.get('format', 'json')
    
# #     if format not in ['json', 'csv', 'excel']:
# #         return jsonify({'error': 'Invalid format specified'}), 400
    
# #     data = tracker.export_data(user_id, format)
# #     if not data:
# #         return jsonify({'error': 'User not found'}), 404
    
# #     filename = f"fitness_data_{user_id}_{datetime.now().strftime('%Y%m%d')}"
    
# #     if format == 'json':
# #         filename += '.json'
# #         return send_file(
# #             StringIO(data),
# #             mimetype='application/json',
# #             as_attachment=True,
# #             download_name=filename
# #         )
# #     elif format == 'csv':
# #         filename += '.csv'
# #         return send_file(
# #             StringIO(data),
# #             mimetype='text/csv',
# #             as_attachment=True,
# #             download_name=filename
# #         )
# #     elif format == 'excel':
# #         filename += '.xlsx'
# #         return send_file(
# #             StringIO(data),
# #             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
# #             as_attachment=True,
# #             download_name=filename
# #         )

# # @app.route('/api/ai/generate', methods=['POST'])
# # def generate_content():
# #     data = request.json
# #     prompt = data.get('prompt')
    
# #     if not prompt:
# #         return jsonify({'error': 'Prompt is required'}), 400
    
# #     try:
# #         response = gemini_model.generate_content(prompt)
# #         return jsonify({'response': response.text})
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # # File upload and processing
# # ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx'}

# # def allowed_file(filename):
# #     return '.' in filename and \
# #            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # @app.route('/api/upload', methods=['POST'])
# # def upload_file():
# #     if 'file' not in request.files:
# #         return jsonify({'error': 'No file part'}), 400
    
# #     file = request.files['file']
# #     if file.filename == '':
# #         return jsonify({'error': 'No selected file'}), 400
    
# #     if file and allowed_file(file.filename):
# #         filename = secure_filename(file.filename)
# #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #         file.save(filepath)
        
# #         try:
# #             # Process the file based on its type
# #             if filename.endswith('.csv'):
# #                 df = pd.read_csv(filepath)
# #             elif filename.endswith('.json'):
# #                 df = pd.read_json(filepath)
# #             elif filename.endswith('.xlsx'):
# #                 df = pd.read_excel(filepath)
# #             else:
# #                 return jsonify({'error': 'Unsupported file format'}), 400
            
# #             # Generate analysis using AI
# #             prompt = f"""
# #             Analyze this fitness data and provide insights:
# #             {df.head().to_string()}
            
# #             Provide:
# #             1. Key statistics
# #             2. Notable trends
# #             3. Recommendations for improvement
# #             """
            
# #             analysis = gemini_model.generate_content(prompt).text
            
# #             return jsonify({
# #                 'message': 'File processed successfully',
# #                 'analysis': analysis,
# #                 'data_preview': df.head().to_dict()
# #             })
# #         except Exception as e:
# #             return jsonify({'error': f'Error processing file: {str(e)}'}), 500
# #         finally:
# #             # Clean up - remove the uploaded file
# #             try:
# #                 os.remove(filepath)
# #             except:
# #                 pass
# #     else:
# #         return jsonify({'error': 'Invalid file type'}), 400

# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5000, debug=True)
import os
from datetime import datetime
import json
import csv
from io import StringIO
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from google.generativeai import GenerativeModel, configure
import requests
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.secret_key = os.getenv('FLASK_SECRET_KEY')

# Configure AI services
configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = GenerativeModel('gemini-2.5-flash')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Database configuration
db_config = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': os.getenv('DB_PORT', '3306')
}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def create_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def initialize_database():
    connection = create_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Create users table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                age INT NOT NULL,
                weight FLOAT NOT NULL,
                height FLOAT NOT NULL,
                fitness_level VARCHAR(255) NOT NULL,
                created_at DATETIME NOT NULL
            )
            """)
            
            # Create workouts table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS workouts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                workout_type VARCHAR(255) NOT NULL,
                duration INT NOT NULL,
                calories FLOAT NOT NULL,
                intensity VARCHAR(255) NOT NULL,
                notes TEXT,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            """)
            
            # Create goals table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                goal_type VARCHAR(255) NOT NULL,
                target_value FLOAT NOT NULL,
                current_value FLOAT DEFAULT 0,
                target_date DATE NOT NULL,
                created_at DATETIME NOT NULL,
                completed BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            """)
            
            connection.commit()
            print("Database tables initialized successfully")
        except Error as e:
            print(f"Error initializing database: {e}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

# Initialize database tables
initialize_database()

class FitnessTracker:
    def __init__(self):
        pass

    def add_user(self, user_id, name, age, weight, height, fitness_level):
        connection = create_db_connection()
        if not connection:
            return False
            
        try:
            cursor = connection.cursor()
            cursor.execute("""
            INSERT INTO users (user_id, name, age, weight, height, fitness_level, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (user_id, name, age, weight, height, fitness_level, datetime.now()))
            connection.commit()
            return True
        except Error as e:
            print(f"Error adding user: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    def log_workout(self, user_id, workout_type, duration, calories, intensity, notes=""):
        connection = create_db_connection()
        if not connection:
            return None
            
        try:
            cursor = connection.cursor()
            cursor.execute("""
            INSERT INTO workouts (user_id, workout_type, duration, calories, intensity, notes, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (user_id, workout_type, duration, calories, intensity, notes, datetime.now()))
            workout_id = cursor.lastrowid
            connection.commit()
            
            # Return the logged workout
            cursor.execute("SELECT * FROM workouts WHERE id = %s", (workout_id,))
            workout = cursor.fetchone()
            
            if workout:
                return {
                    'id': workout[0],
                    'user_id': workout[1],
                    'type': workout[2],
                    'duration': workout[3],
                    'calories': workout[4],
                    'intensity': workout[5],
                    'notes': workout[6],
                    'timestamp': workout[7].isoformat()
                }
            return None
        except Error as e:
            print(f"Error logging workout: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    def set_goal(self, user_id, goal_type, target_value, target_date):
        connection = create_db_connection()
        if not connection:
            return None
            
        try:
            cursor = connection.cursor()
            cursor.execute("""
            INSERT INTO goals (user_id, goal_type, target_value, target_date, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """, (user_id, goal_type, target_value, target_date, datetime.now()))
            goal_id = cursor.lastrowid
            connection.commit()
            
            # Return the set goal
            cursor.execute("SELECT * FROM goals WHERE id = %s", (goal_id,))
            goal = cursor.fetchone()
            
            if goal:
                return {
                    'id': goal[0],
                    'user_id': goal[1],
                    'type': goal[2],
                    'target_value': goal[3],
                    'current_value': goal[4],
                    'target_date': goal[5].isoformat(),
                    'created_at': goal[6].isoformat(),
                    'completed': goal[7]
                }
            return None
        except Error as e:
            print(f"Error setting goal: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
# Add to FitnessTracker class

    def update_goal_progress(self, goal_id, current_value):
        connection = create_db_connection()
        if not connection:
            return None
            
        try:
            cursor = connection.cursor()
            
            # Update current value
            cursor.execute("""
            UPDATE goals 
            SET current_value = %s 
            WHERE id = %s
            """, (current_value, goal_id))
            
            # Check if goal is completed
            cursor.execute("""
            UPDATE goals 
            SET completed = CASE 
                WHEN target_value <= %s THEN TRUE 
                ELSE FALSE 
            END
            WHERE id = %s
            """, (current_value, goal_id))
            
            connection.commit()
            
            # Return the updated goal
            cursor.execute("SELECT * FROM goals WHERE id = %s", (goal_id,))
            goal = cursor.fetchone()
            
            if goal:
                return {
                    'id': goal[0],
                    'user_id': goal[1],
                    'type': goal[2],
                    'target_value': goal[3],
                    'current_value': goal[4],
                    'target_date': goal[5].isoformat(),
                    'created_at': goal[6].isoformat(),
                    'completed': goal[7]
                }
            return None
        except Error as e:
            print(f"Error updating goal progress: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    def search_users(self, search_term):
        connection = create_db_connection()
        if not connection:
            return []
            
        try:
            cursor = connection.cursor(dictionary=True)
            query = """
            SELECT * FROM users 
            WHERE user_id LIKE %s 
            OR name LIKE %s
            """
            cursor.execute(query, (f"%{search_term}%", f"%{search_term}%"))
            return cursor.fetchall()
        except Error as e:
            print(f"Error searching users: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    def get_user_stats(self, user_id):
        connection = create_db_connection()
        if not connection:
            return None
            
        try:
            cursor = connection.cursor(dictionary=True)
            
            # Get user info
            cursor.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            user = cursor.fetchone()
            
            if not user:
                return None
                
            # Convert datetime to string
            user['created_at'] = user['created_at'].isoformat()
            
            # Get workout stats
            cursor.execute("""
            SELECT COUNT(*) as total_workouts, SUM(calories) as total_calories
            FROM workouts
            WHERE user_id = %s
            """, (user_id,))
            workout_stats = cursor.fetchone()
            
            # Get goal stats
            cursor.execute("""
            SELECT 
                SUM(CASE WHEN completed = FALSE THEN 1 ELSE 0 END) as active_goals,
                SUM(CASE WHEN completed = TRUE THEN 1 ELSE 0 END) as completed_goals
            FROM goals
            WHERE user_id = %s
            """, (user_id,))
            goal_stats = cursor.fetchone()
            
            # Get workouts
            cursor.execute("SELECT * FROM workouts WHERE user_id = %s", (user_id,))
            workouts = []
            for workout in cursor.fetchall():
                workout['timestamp'] = workout['timestamp'].isoformat()
                workouts.append(workout)
            
            # Get goals
            cursor.execute("SELECT * FROM goals WHERE user_id = %s", (user_id,))
            goals = []
            for goal in cursor.fetchall():
                goal['target_date'] = goal['target_date'].isoformat()
                goal['created_at'] = goal['created_at'].isoformat()
                goals.append(goal)
            
            stats = {
                'user_info': user,
                'total_workouts': workout_stats['total_workouts'] or 0,
                'total_calories': workout_stats['total_calories'] or 0,
                'active_goals': goal_stats['active_goals'] or 0,
                'completed_goals': goal_stats['completed_goals'] or 0,
                'workouts': workouts,
                'goals': goals
            }
            return stats
        except Error as e:
            print(f"Error getting user stats: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    def get_workout_recommendation(self, user_id):
        stats = self.get_user_stats(user_id)
        if not stats:
            return None
            
        user = stats['user_info']
        prompt = f"""
        Based on the following user profile, suggest 3 personalized workout routines:
        - Name: {user['name']}
        - Age: {user['age']}
        - Weight: {user['weight']} kg
        - Height: {user['height']} cm
        - Fitness Level: {user['fitness_level']}
        
        The user has completed {stats['total_workouts']} workouts so far.
        Provide recommendations that consider their fitness level and would help them progress.
        Format the response as a markdown list with brief explanations for each recommendation.
        """
        
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return None

    def analyze_workout_sentiment(self, workout_notes):
        if not workout_notes:
            return None
        
        API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": workout_notes})
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                sentiment = result[0][0]['label']
                return sentiment
            return None
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return None

    def export_data(self, user_id, format='json'):
        stats = self.get_user_stats(user_id)
        if not stats:
            return None
        
        data = {
            'user': stats['user_info'],
            'workouts': stats['workouts'],
            'goals': stats['goals']
        }
        
        if format == 'json':
            return json.dumps(data, indent=2)
        elif format == 'csv':
            output = StringIO()
            writer = csv.writer(output)
            
            # Write user info
            writer.writerow(['User Information'])
            writer.writerow(['Name', 'Age', 'Weight', 'Height', 'Fitness Level'])
            writer.writerow([
                data['user']['name'],
                data['user']['age'],
                data['user']['weight'],
                data['user']['height'],
                data['user']['fitness_level']
            ])
            
            # Write workouts
            writer.writerow([])
            writer.writerow(['Workouts'])
            writer.writerow(['ID', 'Type', 'Duration', 'Calories', 'Intensity', 'Notes', 'Timestamp'])
            for workout in data['workouts']:
                writer.writerow([
                    workout['id'],
                    workout['workout_type'],
                    workout['duration'],
                    workout['calories'],
                    workout['intensity'],
                    workout['notes'],
                    workout['timestamp']
                ])
            
            # Write goals
            writer.writerow([])
            writer.writerow(['Goals'])
            writer.writerow(['ID', 'Type', 'Target Value', 'Current Value', 'Target Date', 'Completed'])
            for goal in data['goals']:
                writer.writerow([
                    goal['id'],
                    goal['goal_type'],
                    goal['target_value'],
                    goal['current_value'],
                    goal['target_date'],
                    goal['completed']
                ])
            
            return output.getvalue()
        elif format == 'excel':
            # Prepare data for Excel
            user_data = {
                'Name': [data['user']['name']],
                'Age': [data['user']['age']],
                'Weight': [data['user']['weight']],
                'Height': [data['user']['height']],
                'Fitness Level': [data['user']['fitness_level']]
            }
            df_user = pd.DataFrame(user_data)
            
            workouts_data = []
            for workout in data['workouts']:
                workouts_data.append({
                    'ID': workout['id'],
                    'Type': workout['workout_type'],
                    'Duration': workout['duration'],
                    'Calories': workout['calories'],
                    'Intensity': workout['intensity'],
                    'Notes': workout['notes'],
                    'Timestamp': workout['timestamp']
                })
            df_workouts = pd.DataFrame(workouts_data)
            
            goals_data = []
            for goal in data['goals']:
                goals_data.append({
                    'ID': goal['id'],
                    'Type': goal['goal_type'],
                    'Target Value': goal['target_value'],
                    'Current Value': goal['current_value'],
                    'Target Date': goal['target_date'],
                    'Completed': goal['completed']
                })
            df_goals = pd.DataFrame(goals_data)
            
            output = StringIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_user.to_excel(writer, sheet_name='User Info', index=False)
                df_workouts.to_excel(writer, sheet_name='Workouts', index=False)
                df_goals.to_excel(writer, sheet_name='Goals', index=False)
            
            return output.getvalue()
        else:
            return None
    def get_workout_analytics(self, user_id, period='month'):
        connection = create_db_connection()
        if not connection:
            return None
            
        try:
            cursor = connection.cursor(dictionary=True)
            
            # Determine date range based on period
            if period == 'week':
                date_format = "%Y-%u"  # Year-week
                interval = "7 DAY"
            elif period == 'month':
                date_format = "%Y-%m"   # Year-month
                interval = "1 MONTH"
            else:  # year
                date_format = "%Y"     # Year
                interval = "1 YEAR"
            
            # Get workout frequency by type
            cursor.execute(f"""
            SELECT 
                workout_type,
                COUNT(*) as count,
                SUM(duration) as total_duration,
                SUM(calories) as total_calories
            FROM workouts
            WHERE user_id = %s
            AND timestamp >= DATE_SUB(NOW(), INTERVAL {interval})
            GROUP BY workout_type
            ORDER BY count DESC
            """, (user_id,))
            by_type = cursor.fetchall()
            
            # Get weekly/monthly trends
            cursor.execute(f"""
            SELECT 
                DATE_FORMAT(timestamp, %s) as period,
                COUNT(*) as count,
                SUM(duration) as total_duration,
                SUM(calories) as total_calories
            FROM workouts
            WHERE user_id = %s
            AND timestamp >= DATE_SUB(NOW(), INTERVAL {interval})
            GROUP BY period
            ORDER BY period
            """, (date_format, user_id))
            trends = cursor.fetchall()
            
            # Get intensity distribution
            cursor.execute(f"""
            SELECT 
                intensity,
                COUNT(*) as count,
                AVG(duration) as avg_duration,
                AVG(calories) as avg_calories
            FROM workouts
            WHERE user_id = %s
            AND timestamp >= DATE_SUB(NOW(), INTERVAL {interval})
            GROUP BY intensity
            """, (user_id,))
            intensity = cursor.fetchall()
            
            return {
                'by_type': by_type,
                'trends': trends,
                'intensity': intensity
            }
        except Error as e:
            print(f"Error getting workout analytics: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    def add_social_features(self):
        connection = create_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                
                # Create challenges table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS challenges (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    start_date DATETIME NOT NULL,
                    end_date DATETIME NOT NULL,
                    target_value FLOAT NOT NULL,
                    metric VARCHAR(50) NOT NULL,
                    created_by VARCHAR(255) NOT NULL,
                    created_at DATETIME NOT NULL
                )
                """)
                
                # Create user_challenges table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_challenges (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    challenge_id INT NOT NULL,
                    current_value FLOAT DEFAULT 0,
                    joined_at DATETIME NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (challenge_id) REFERENCES challenges(id)
                )
                """)
                
                connection.commit()
                print("Social features tables initialized successfully")
            except Error as e:
                print(f"Error initializing social features: {e}")
            finally:
                if connection.is_connected():
                    cursor.close()
                    connection.close()

    def generate_personalized_meal_plan(self, user_id):
        stats = self.get_user_stats(user_id)
        if not stats:
            return None
            
        user = stats['user_info']
        prompt = f"""
        Generate a personalized 7-day meal plan for this user:
        - Name: {user['name']}
        - Age: {user['age']}
        - Weight: {user['weight']} kg
        - Height: {user['height']} cm
        - Fitness Level: {user['fitness_level']}
        
        The user has burned {stats['total_calories']} calories in {stats['total_workouts']} workouts.
        Consider their fitness goals from the database.
        
        Format as:
        - Day 1:
          * Breakfast: [meal] - [calories] calories
          * Lunch: [meal] - [calories] calories
          * Dinner: [meal] - [calories] calories
          * Snacks: [snack] - [calories] calories
        - Day 2: ...
        
        Include macronutrient breakdown for each day.
        """
        
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating meal plan: {e}")
            return None
# Initialize tracker
tracker = FitnessTracker()

# Routes (remain the same as in your original code)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/users', methods=['POST'])
def add_user():
    data = request.json
    required_fields = ['user_id', 'name', 'age', 'weight', 'height', 'fitness_level']
    
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    success = tracker.add_user(
        data['user_id'],
        data['name'],
        data['age'],
        data['weight'],
        data['height'],
        data['fitness_level']
    )
    
    if not success:
        return jsonify({'error': 'Failed to add user'}), 500
    
    return jsonify({'message': 'User added successfully'}), 201

@app.route('/api/users/<user_id>/workouts', methods=['POST'])
def log_workout(user_id):
    data = request.json
    required_fields = ['workout_type', 'duration', 'calories', 'intensity']
    
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    notes = data.get('notes', '')
    workout = tracker.log_workout(
        user_id,
        data['workout_type'],
        data['duration'],
        data['calories'],
        data['intensity'],
        notes
    )
    
    if not workout:
        return jsonify({'error': 'Failed to log workout'}), 500
    
    # Analyze sentiment if notes are provided
    sentiment = None
    if notes:
        sentiment = tracker.analyze_workout_sentiment(notes)
    
    response = {
        'message': 'Workout logged successfully',
        'workout': workout,
        'sentiment': sentiment
    }
    return jsonify(response), 201

@app.route('/api/users/<user_id>/goals', methods=['POST'])
def set_goal(user_id):
    data = request.json
    required_fields = ['goal_type', 'target_value', 'target_date']
    
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    goal = tracker.set_goal(
        user_id,
        data['goal_type'],
        data['target_value'],
        data['target_date']
    )
    
    if not goal:
        return jsonify({'error': 'Failed to set goal'}), 500
    
    return jsonify({'message': 'Goal set successfully', 'goal': goal}), 201

@app.route('/api/users/<user_id>/stats', methods=['GET'])
def get_stats(user_id):
    stats = tracker.get_user_stats(user_id)
    if not stats:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(stats)
@app.route('/api/users/<user_id>/goals/<int:goal_id>', methods=['PUT'])
def update_goal_progress(user_id, goal_id):
    data = request.json
    if 'current_value' not in data:
        return jsonify({'error': 'Missing current_value'}), 400
    
    updated_goal = tracker.update_goal_progress(goal_id, data['current_value'])
    if not updated_goal:
        return jsonify({'error': 'Failed to update goal progress'}), 500
    
    return jsonify({'message': 'Goal progress updated', 'goal': updated_goal})

@app.route('/api/users/<user_id>/analytics', methods=['GET'])
def get_analytics(user_id):
    period = request.args.get('period', 'month')
    analytics = tracker.get_workout_analytics(user_id, period)
    if not analytics:
        return jsonify({'error': 'Unable to generate analytics'}), 500
    return jsonify(analytics)

@app.route('/api/users/search', methods=['GET'])
def search_users():
    search_term = request.args.get('q')
    if not search_term:
        return jsonify({'error': 'Missing search term'}), 400
    
    users = tracker.search_users(search_term)
    return jsonify(users)

@app.route('/api/users/<user_id>/meal-plan', methods=['GET'])
def get_meal_plan(user_id):
    meal_plan = tracker.generate_personalized_meal_plan(user_id)
    if not meal_plan:
        return jsonify({'error': 'Unable to generate meal plan'}), 500
    return jsonify({'meal_plan': meal_plan})
@app.route('/api/users/<user_id>/recommendations', methods=['GET'])
def get_recommendations(user_id):
    recommendation = tracker.get_workout_recommendation(user_id)
    if not recommendation:
        return jsonify({'error': 'Unable to generate recommendations'}), 500
    return jsonify({'recommendation': recommendation})

@app.route('/api/users/<user_id>/export', methods=['GET'])
def export_data(user_id):
    format = request.args.get('format', 'json')
    
    if format not in ['json', 'csv', 'excel']:
        return jsonify({'error': 'Invalid format specified'}), 400
    
    data = tracker.export_data(user_id, format)
    if not data:
        return jsonify({'error': 'User not found'}), 404
    
    filename = f"fitness_data_{user_id}_{datetime.now().strftime('%Y%m%d')}"
    
    if format == 'json':
        filename += '.json'
        return send_file(
            StringIO(data),
            mimetype='application/json',
            as_attachment=True,
            download_name=filename
        )
    elif format == 'csv':
        filename += '.csv'
        return send_file(
            StringIO(data),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    elif format == 'excel':
        filename += '.xlsx'
        return send_file(
            StringIO(data),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )

@app.route('/api/ai/generate', methods=['POST'])
def generate_content():
    data = request.json
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    try:
        response = gemini_model.generate_content(prompt)
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# File upload and processing
ALLOWED_EXTENSIONS = {'csv', 'json', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the file based on its type
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith('.json'):
                # For JSON files, load the data first
                with open(filepath) as f:
                    data = json.load(f)
                
                # Create separate DataFrames for each section
                user_df = pd.DataFrame([{k:v for k,v in data.items() if not isinstance(v, list)}])
                daily_df = pd.DataFrame(data['daily_summary'])
                diet_df = pd.json_normalize(data['diet_log'], record_path='meals', meta='date')
                exercise_df = pd.DataFrame(data['exercise_log'])
                
                # Store all DataFrames in a dictionary
                dfs = {
                    'user_info': user_df,
                    'daily_summary': daily_df,
                    'diet_log': diet_df,
                    'exercise_log': exercise_df
                }
                
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
            
            # Generate analysis using AI
            prompt = f"""
            Analyze this fitness data and provide insights:
            User Information: {dfs['user_info'].to_string()}
            Daily Summary: {dfs['daily_summary'].to_string()}
            Diet Log: {dfs['diet_log'].to_string()}
            Exercise Log: {dfs['exercise_log'].to_string()}
            
            Provide:
            1. Key statistics
            2. Notable trends
            3. Recommendations for improvement based on the user's weight loss goal
            """
            
            analysis = gemini_model.generate_content(prompt).text
            
            return jsonify({
                'message': 'File processed successfully',
                'analysis': analysis,
                'data_preview': {
                    'user_info': dfs['user_info'].to_dict('records')[0],
                    'daily_summary': dfs['daily_summary'].head().to_dict('records'),
                    'diet_log': dfs['diet_log'].head().to_dict('records'),
                    'exercise_log': dfs['exercise_log'].head().to_dict('records')
                }
            })
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400
# @app.route('/api/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         try:
#             # Process the file based on its type
#             if filename.endswith('.csv'):
#                 df = pd.read_csv(filepath)
#             elif filename.endswith('.json'):
#                 df = pd.read_json(filepath)
#             elif filename.endswith('.xlsx'):
#                 df = pd.read_excel(filepath)
#             else:
#                 return jsonify({'error': 'Unsupported file format'}), 400
            
#             # Generate analysis using AI
#             prompt = f"""
#             Analyze this fitness data and provide insights:
#             {df.head().to_string()}
            
#             Provide:
#             1. Key statistics
#             2. Notable trends
#             3. Recommendations for improvement
#             """
            
#             analysis = gemini_model.generate_content(prompt).text
            
#             return jsonify({
#                 'message': 'File processed successfully',
#                 'analysis': analysis,
#                 'data_preview': df.head().to_dict()
#             })
#         except Exception as e:
#             return jsonify({'error': f'Error processing file: {str(e)}'}), 500
#     else:
#         return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)