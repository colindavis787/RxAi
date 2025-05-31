from flask import Flask, render_template, request, redirect, url_for, session, flash
import yaml
from yaml.loader import SafeLoader
import bcrypt
import subprocess
import os

app = Flask(__name__)
app.secret_key = '2f0782073d00457d2c4ed7576e6771c8'  # Your secure key

# Load credentials
with open('.streamlit/credentials.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Extract credentials for verification
users = config['credentials']['usernames']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if username in users and bcrypt.checkpw(password.encode('utf-8'), users[username]['password'].encode('utf-8')):
            session['authentication_status'] = True
            session['username'] = username
            session['name'] = users[username]['name']
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if session.get('authentication_status'):
        return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('authentication_status', None)
    session.pop('username', None)
    session.pop('name', None)
    return redirect(url_for('index'))

@app.route('/streamlit')
def streamlit_app():
    if session.get('authentication_status'):
        # Run Streamlit app in a subprocess
        subprocess.Popen(['streamlit', 'run', '../app.py', '--server.port', '8502'])
        return redirect('http://localhost:8502')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)