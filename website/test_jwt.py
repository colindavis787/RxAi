import jwt
import datetime

payload = {
    'username': 'test13',
    'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
}
token = jwt.encode(payload, 'your_jwt_secret_key_12345', algorithm='HS256')
token = token.decode('utf-8') if isinstance(token, bytes) else token
segments = token.split('.')
print(f"Token: {token}")
print(f"Segments: {len(segments)}")
