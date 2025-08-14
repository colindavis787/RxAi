from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
import sqlite3
import bcrypt
import os
import logging
import jwt
import datetime

# Disable watchdog logging
logging.getLogger('watchdog').setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='website/static', template_folder='website/templates')
app.secret_key = os.getenv('FLASK_SECRET_KEY', '2f0782073d00457d2c4ed7576e6771c8')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your_jwt_secret_key_12345')

# Database path (relative to app location, adjust for EB if using a different DB)
db_path = os.path.join(os.path.dirname(__file__), 'users.db')

def get_db_connection():
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}", exc_info=True)
        raise

def load_users():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT username, name, password FROM users")
        users = {row['username']: {'name': row['name'], 'password': row['password']} for row in cursor.fetchall()}
        conn.close()
        logger.debug(f"Loaded users from database: {list(users.keys())}")
        return users
    except Exception as e:
        logger.error(f"Failed to load users from database: {str(e)}", exc_info=True)
        return {}

@app.route('/health')
def health():
    return "OK", 200

@app.route('/')
def index():
    try:
        logger.debug("Attempting to render index.html")
        result = render_template('index.html')
        logger.debug("Successfully rendered index.html")
        return result
    except Exception as e:
        logger.error(f"Error rendering homepage: {str(e)}", exc_info=True)
        return Response(f"Error rendering homepage: {str(e)}", status=500)

@app.route('/login', methods=['GET', 'POST'])
def login():
    logger.debug("Accessing login route")
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'templates', 'login.html')):
        logger.error("login.html not found in templates directory")
        return Response("Template login.html not found", status=500)
    users = load_users()
    if not users:
        logger.error("No users loaded from database")
        flash('Authentication system is unavailable. Please contact support.', 'danger')
        return render_template('login.html')
    logger.debug(f"Request method: {request.method}")
    if request.method == 'POST':
        logger.debug("Received POST request for login")
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        logger.debug(f"Form data received - Username: {username}, Password: {'[REDACTED]' if password else 'None'}")
        if not username or not password:
            logger.warning("Missing username or password")
            flash('Username and password are required.', 'danger')
            logger.debug("Rendering login.html due to missing fields")
            return render_template('login.html')
        if len(username) > 50 or len(password) > 50:
            logger.warning("Login input too long")
            flash('Input is too long (max 50 characters).', 'danger')
            logger.debug("Rendering login.html due to input length")
            return render_template('login.html')
        if username not in users:
            logger.warning(f"Username {username} not found")
            flash('Invalid username or password.', 'danger')
            logger.debug("Rendering login.html due to invalid username")
            return render_template('login.html')
        stored_password = users[username]['password']
        logger.debug(f"Stored password hash: {stored_password}")
        try:
            if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
                session['authentication_status'] = True
                session['username'] = username
                session['name'] = users[username]['name']
                token = jwt.encode({
                    'username': username,
                    'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
                }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
                # Ensure token is a string
                token = token.decode('utf-8') if isinstance(token, bytes) else token
                session['token'] = token
                logger.info(f"Token generated for {username}: {token}")
                flash('Login successful! Welcome to your dashboard.', 'success')
                logger.info(f"Successful login for {username}")
                logger.debug(f"Redirecting to dashboard for {username}")
                return redirect(url_for('dashboard'))
            else:
                logger.warning("Invalid password")
                flash('Invalid username or password.', 'danger')
                logger.debug("Rendering login.html due to invalid password")
                return render_template('login.html')
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            flash(f'Login error: {str(e)}', 'danger')
            logger.debug("Rendering login.html due to login exception")
            return render_template('login.html')
    logger.debug("Rendering login.html for GET request")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    logger.debug("Accessing register route")
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'templates', 'register.html')):
        logger.error("register.html not found in templates directory")
        return Response("Template register.html not found", status=500)
    logger.debug(f"Request method: {request.method}")
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        logger.debug(f"Form data received - Name: {name}, Username: {username}, Password: {'[REDACTED]' if password else 'None'}")
        if not name or not username or not password:
            logger.warning("Missing registration fields")
            flash('All fields are required.', 'danger')
            logger.debug("Rendering register.html due to missing fields")
            return render_template('register.html')
        if len(name) > 50 or len(username) > 50 or len(password) > 50:
            logger.warning("Registration input too long")
            flash('Input is too long (max 50 characters).', 'danger')
            logger.debug("Rendering register.html due to input length")
            return render_template('register.html')
        users = load_users()
        if username in users:
            logger.warning(f"Username {username} already exists")
            flash('Username already exists.', 'danger')
            logger.debug("Rendering register.html due to existing username")
            return render_template('register.html')
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        # Insert the new user into the database
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, name, password) VALUES (?, ?, ?)',
                           (username, name, hashed_password))
            conn.commit()
            logger.info(f"Successfully registered user: {username}")
            flash('Registration successful! Please log in.', 'success')
            logger.debug(f"Redirecting to login for {username}")
        except Exception as e:
            logger.error(f"Failed to register user: {str(e)}")
            flash('Registration failed. Please try again.', 'danger')
            logger.debug("Rendering register.html due to registration failure")
        finally:
            conn.close()
        return redirect(url_for('login'))
    logger.debug("Rendering register.html for GET request")
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    logger.debug("Accessing dashboard route")
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'templates', 'dashboard.html')):
        logger.error("dashboard.html not found in templates directory")
        return Response("Template dashboard.html not found", status=500)
    if not session.get('authentication_status'):
        logger.warning("Unauthorized dashboard access, redirecting to login")
        return redirect(url_for('login'))
    if not session.get('username') or not session.get('token'):
        logger.error("Missing username or token in session, redirecting to login")
        flash('Session expired or invalid. Please log in again.', 'danger')
        return redirect(url_for('login'))
    streamlit_url = os.getenv('STREAMLIT_URL', f"https://q9dhs7s8xfly3gtvwuwpfm.streamlit.app/?embedded=true&username={session.get('username', '')}&token={session.get('token', '')}")
    logger.debug(f"Streamlit URL: {streamlit_url}")
    try:
        import requests
        response = requests.head(streamlit_url, timeout=5)
        if response.status_code != 200:
            logger.warning(f"Streamlit app unavailable at {streamlit_url}, status code: {response.status_code}")
            flash('Streamlit app is currently unavailable. Please try again later.', 'danger')
    except requests.RequestException as e:
        logger.error(f"Failed to reach Streamlit app: {str(e)}")
        flash('Streamlit app is currently unavailable. Please try again later.', 'danger')
    return render_template('dashboard.html', username=session['username'], streamlit_url=streamlit_url)

@app.route('/logout', methods=['POST'])
def logout():
    try:
        logger.debug("Accessing logout route")
        session.clear()
        logger.info("User logged out successfully")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        return Response(f"Error during logout: {str(e)}", status=500)

@app.route('/analyze', methods=['POST'])
def analyze():
    if not session.get('authentication_status'):
        return redirect(url_for('login'))
    file = request.files.get('file')
    if not file:
        flash('No file uploaded.', 'danger')
        return redirect(url_for('dashboard'))
    temp_path = '/tmp/' + file.filename
    file.save(temp_path)
    from pharmacy_analyzer import main
    df, messages, analysis_results, anomalies, chart_files, predictions = main(temp_path, inflation_rate=0.05)
    os.remove(temp_path)
    if df is None:
        flash(messages, 'danger')
        return redirect(url_for('dashboard'))
    return render_template('analysis.html', messages=messages, analysis_results=analysis_results, 
                           anomalies=anomalies.to_html() if not anomalies.empty else "No anomalies detected",
                           chart_files=chart_files, predictions=predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=False)  # Production-ready; debug disabled
