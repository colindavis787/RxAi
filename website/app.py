from flask import Flask, render_template, request, redirect, url_for, session, flash
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader
    import subprocess
    import os

    app = Flask(__name__)
    app.secret_key = 'your_secret_key_here'  # Replace with a secure random key

    # Load credentials
    with open('.streamlit/credentials.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            # Simulate streamlit-authenticator login
            try:
                authenticator.login(location='main')
                if authenticator._check_credentials(username, password):
                    session['authentication_status'] = True
                    session['username'] = username
                    session['name'] = config['credentials']['usernames'][username]['name']
                    return redirect(url_for('dashboard'))
                else:
                    flash('Invalid username or password', 'error')
            except:
                flash('Login error', 'error')
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