from flask import Flask, render_template, request, redirect, url_for, session, flash
import yaml
from yaml.loader import SafeLoader
import bcrypt
import subprocess
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
app.secret_key = '2f0782073d00457d2c4ed7576e6771c8'

# Load credentials
with open('.streamlit/credentials.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Extract credentials for verification
users = config['credentials']['usernames']

@app.route('/')
def index():
    try:
        logger.debug("Rendering index.html")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering homepage: {str(e)}")
        return f"Error rendering homepage: {str(e)}", 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        logger.debug("Accessing login route")
        if request.method == 'POST':
            username = request.form['username'].strip()
            password = request.form['password']
            logger.debug(f"Login attempt for username: {username}")
            if len(username) > 50 or len(password) > 50:
                flash('Input too long', 'error')
                logger.warning("Login input too long")
                return render_template('login.html')
            if not username or not password:
                flash('Username and password required', 'error')
                logger.warning("Missing username or password")
                return render_template('login.html')
            if username in users and bcrypt.checkpw(password.encode('utf-8'), users[username]['password'].encode('utf-8')):
                session['authentication_status'] = True
                session['username'] = username
                session['name'] = users[username]['name']
                logger.info(f"Successful login for {username}")
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password', 'error')
                logger.warning("Invalid login credentials")
        return render_template('login.html')
    except Exception as e:
        logger.error(f"Error in login route: {str(e)}")
        return f"Error in login: {str(e)}", 500

@app.route('/dashboard')
def dashboard():
    try:
        logger.debug("Accessing dashboard route")
        if session.get('authentication_status'):
            return render_template('dashboard.html', username=session['username'])
        return redirect(url_for('login'))
    except Exception as e:
        logger.error(f"Error in dashboard route: {str(e)}")
        return f"Error in dashboard: {str(e)}", 500

@app.route('/logout', methods=['POST'])
def logout():
    try:
        logger.debug("Logging out")
        session.pop('authentication_status', None)
        session.pop('username', None)
        session.pop('name', None)
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error in logout route: {str(e)}")
        return f"Error in logout: {str(e)}", 500

@app.route('/streamlit')
def streamlit_app():
    try:
        logger.debug("Accessing streamlit route")
        if session.get('authentication_status'):
            # Run Streamlit app in a subprocess
            subprocess.Popen(['streamlit', 'run', '../app.py', '--server.port', '8502'])
            logger.debug("Started Streamlit subprocess")
            return redirect('http://localhost:8502')
        return redirect(url_for('login'))
    except Exception as e:
        logger.error(f"Error in streamlit route: {str(e)}")
        return f"Error in streamlit: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)