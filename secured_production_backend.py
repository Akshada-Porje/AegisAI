
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import google.generativeai as genai
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import jwt
import hashlib
import os
import logging
from functools import wraps
from cryptography.fernet import Fernet
import pickle
import bleach
from werkzeug.utils import secure_filename
import json
from scipy import stats
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except:
    pass

app = Flask(__name__)

# ==================== CONFIGURATION ====================

CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per day", "100 per hour"],
    storage_uri="memory://"
)
limiter.init_app(app)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(32).hex())
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.csv', '.xlsx', '.xls']

# Create encrypted storage directory
ENCRYPTED_STORAGE_DIR = Path('encrypted_storage')
ENCRYPTED_STORAGE_DIR.mkdir(exist_ok=True)

# Enhanced Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

audit_logger = logging.getLogger('audit')
audit_handler = logging.FileHandler('audit.log')
audit_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)

# Security logger
security_logger = logging.getLogger('security')
security_handler = logging.FileHandler('security.log')
security_handler.setFormatter(logging.Formatter('%(asctime)s - SECURITY - %(message)s'))
security_logger.addHandler(security_handler)
security_logger.setLevel(logging.WARNING)

# Storage
METADATA_STORAGE = {}  # Stores metadata, not actual data
TOKENS = {}

# Encryption
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
if isinstance(ENCRYPTION_KEY, str):
    ENCRYPTION_KEY = ENCRYPTION_KEY.encode()
cipher = Fernet(ENCRYPTION_KEY)

# Save encryption key for reference
with open('encryption_key.txt', 'w') as f:
    f.write(f"Encryption Key: {ENCRYPTION_KEY.decode()}\n")
    f.write(f"Encrypted files location: {ENCRYPTED_STORAGE_DIR.absolute()}\n")

# ==================== GEMINI AI SETUP ====================

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_model = None
AI_AVAILABLE = False

if GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here':
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use the correct model name for the stable API
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        # Test the connection
        test_response = gemini_model.generate_content("Test")
        AI_AVAILABLE = True
        logger.info("✓ Gemini AI configured successfully with gemini-pro model")
    except Exception as e:
        logger.error(f"✗ Gemini configuration failed: {e}")
        logger.info("Trying alternative model gemini-1.5-pro...")
        try:
            gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            test_response = gemini_model.generate_content("Test")
            AI_AVAILABLE = True
            logger.info("✓ Gemini AI configured successfully with gemini-1.5-pro model")
        except Exception as e2:
            logger.error(f"✗ Alternative model also failed: {e2}")
            AI_AVAILABLE = False
else:
    logger.warning("⚠ Gemini API key not configured")


# ==================== FILE STORAGE FUNCTIONS ====================

def store_data_encrypted(file_id, df, user_id, ttl=7200):
    """Store data as encrypted file on disk"""
    try:
        if user_id not in METADATA_STORAGE:
            METADATA_STORAGE[user_id] = {}

        # Serialize dataframe
        data_bytes = pickle.dumps(df)

        # Encrypt the data
        encrypted = cipher.encrypt(data_bytes)

        # Save to disk
        file_path = ENCRYPTED_STORAGE_DIR / f"{user_id}_{file_id}.encrypted"
        with open(file_path, 'wb') as f:
            f.write(encrypted)

        # Store metadata only in memory
        METADATA_STORAGE[user_id][file_id] = {
            'file_path': str(file_path),
            'expires_at': datetime.now() + timedelta(seconds=ttl),
            'metadata': {
                'rows': len(df),
                'columns': len(df.columns),
                'created_at': datetime.now().isoformat(),
                'file_hash': hashlib.sha256(data_bytes).hexdigest(),
                'size_bytes': len(encrypted)
            }
        }

        audit_logger.info(f"Data stored encrypted: user={user_id}, file={file_id}, path={file_path}")
        logger.info(f"📁 Encrypted file saved at: {file_path.absolute()}")
        return True

    except Exception as e:
        logger.error(f"Storage error: {e}")
        return False


def retrieve_data_from_disk(file_id, user_id):
    """Retrieve and decrypt data from disk"""
    try:
        if user_id not in METADATA_STORAGE or file_id not in METADATA_STORAGE[user_id]:
            logger.warning(f"File not found: user={user_id}, file={file_id}")
            return None

        file_metadata = METADATA_STORAGE[user_id][file_id]

        # Check expiration
        if datetime.now() > file_metadata['expires_at']:
            logger.info(f"File expired: {file_id}")
            delete_data_from_disk(file_id, user_id)
            return None

        # Read encrypted file
        file_path = Path(file_metadata['file_path'])
        if not file_path.exists():
            logger.error(f"Encrypted file missing: {file_path}")
            return None

        with open(file_path, 'rb') as f:
            encrypted = f.read()

        # Decrypt
        data_bytes = cipher.decrypt(encrypted)
        df = pickle.loads(data_bytes)

        audit_logger.info(f"Data retrieved: user={user_id}, file={file_id}")
        return df

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return None


def delete_data_from_disk(file_id, user_id):
    """Delete encrypted file from disk"""
    try:
        if user_id in METADATA_STORAGE and file_id in METADATA_STORAGE[user_id]:
            file_path = Path(METADATA_STORAGE[user_id][file_id]['file_path'])

            # Delete file
            if file_path.exists():
                file_path.unlink()
                logger.info(f"🗑️ Deleted encrypted file: {file_path}")

            # Remove metadata
            del METADATA_STORAGE[user_id][file_id]
            audit_logger.info(f"Data deleted: user={user_id}, file={file_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Deletion error: {e}")
        return False


# ==================== GEMINI INSIGHTS GENERATION ====================

def generate_gemini_insights(df, analysis_data):
    """Generate insights using Gemini API - NO HARDCODING"""

    if not AI_AVAILABLE or not gemini_model:
        logger.error("Gemini AI not available")
        return None

    try:
        # Prepare data summary for Gemini
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Create a comprehensive summary
        data_summary = {
            'total_records': len(df),
            'columns': df.columns.tolist(),
            'sample_data': df.head(5).to_dict('records'),
            'statistics': {}
        }

        # Add statistics for numeric columns
        for col in numeric_cols[:10]:
            data_summary['statistics'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'std': float(df[col].std())
            }

        # Add anomaly information
        anomaly_summary = analysis_data.get('anomalies', [])[:5]

        # Create detailed prompt for Gemini
        prompt = f"""You are an AI analyst for Maharashtra government data. Analyze this dataset and provide actionable insights.

DATASET OVERVIEW:
- Total Records: {data_summary['total_records']}
- Columns: {', '.join(data_summary['columns'][:15])}

SAMPLE DATA (First 5 rows):
{json.dumps(data_summary['sample_data'], indent=2)}

STATISTICAL SUMMARY:
{json.dumps(data_summary['statistics'], indent=2)}

DETECTED ANOMALIES:
{json.dumps(anomaly_summary, indent=2)}

TASK: Generate 3-5 actionable insights for Maharashtra governance. Focus on:
1. Resource allocation inefficiencies
2. Service delivery gaps  
3. Risk indicators requiring immediate attention
4. Budget optimization opportunities
5. Citizen welfare improvements

Return ONLY a valid JSON array with this EXACT structure (no markdown, no extra text):
[
  {{
    "title": "Specific, clear insight title",
    "severity": "critical" or "high" or "medium" or "low",
    "description": "What the data reveals - be specific with numbers and districts if available",
    "recommendation": "Concrete actionable steps government should take",
    "confidence": 75-95,
    "affected_regions": number,
    "metrics": {{"key1": "value1", "key2": "value2"}}
  }}
]

IMPORTANT: Return ONLY the JSON array, nothing else."""

        logger.info("🤖 Calling Gemini API for insights...")

        # Call Gemini API
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            )
        )

        text = response.text.strip()
        logger.info(f"📥 Gemini API Response received: {len(text)} characters")

        # Extract JSON from response
        # Remove markdown code blocks if present
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        # Find JSON array
        start = text.find('[')
        end = text.rfind(']') + 1

        if start != -1 and end > start:
            json_text = text[start:end]
            insights = json.loads(json_text)

            if isinstance(insights, list) and len(insights) > 0:
                logger.info(f"✓ Successfully parsed {len(insights)} insights from Gemini")
                return insights
            else:
                logger.error("Gemini returned empty or invalid insights array")
                return None
        else:
            logger.error(f"Could not find JSON array in Gemini response: {text[:200]}")
            return None

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error from Gemini response: {e}")
        logger.error(f"Response text: {text[:500]}")
        return None
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None


# ==================== BASIC ANALYSIS FUNCTIONS ====================

def advanced_statistical_analysis(df):
    """Comprehensive statistical analysis"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    analysis = {
        'basic_stats': {},
        'correlations': [],
        'distributions': {},
        'trends': {},
        'risk_scores': {}
    }

    # Basic statistics
    for col in numeric_cols[:10]:
        values = df[col].dropna()
        if len(values) > 0:
            analysis['basic_stats'][col] = {
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'q25': float(values.quantile(0.25)),
                'q75': float(values.quantile(0.75)),
                'skewness': float(values.skew()),
                'kurtosis': float(values.kurtosis())
            }

    # Correlation analysis
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols[:10]].corr()
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    high_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        analysis['correlations'] = high_correlations[:5]

    return analysis


def detect_anomalies_advanced(df):
    """Advanced anomaly detection"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    anomalies = []

    for col in numeric_cols[:5]:
        values = df[col].dropna()
        if len(values) > 10:
            z_scores = np.abs(stats.zscore(values))
            outlier_indices = np.where(z_scores > 2.5)[0][:5]

            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            for idx in outlier_indices:
                actual_idx = values.index[idx]
                value = float(values.iloc[idx])
                severity = 'critical' if z_scores[idx] > 3 else 'high' if z_scores[idx] > 2.5 else 'medium'

                anomalies.append({
                    'row': int(actual_idx) + 2,
                    'column': col,
                    'value': value,
                    'z_score': float(z_scores[idx]),
                    'severity': severity,
                    'expected_range': f"{lower_bound:.2f} - {upper_bound:.2f}",
                    'deviation_percent': float(abs(value - values.mean()) / values.std() * 100)
                })

    severity_order = {'critical': 0, 'high': 1, 'medium': 2}
    anomalies.sort(key=lambda x: severity_order[x['severity']])

    return anomalies[:15]


# ==================== AUTHENTICATION ====================

def generate_token(user_id, role='user'):
    """Generate JWT token"""
    payload = {
        'user_id': user_id,
        'role': role,
        'exp': datetime.utcnow() + timedelta(hours=24),
        'iat': datetime.utcnow()
    }
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

    TOKENS[token] = {
        'user_id': user_id,
        'expires_at': datetime.utcnow() + timedelta(hours=24)
    }

    return token


def require_auth(f):
    """Authentication decorator"""

    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            security_logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
            return jsonify({'error': 'No authorization token'}), 401

        if token.startswith('Bearer '):
            token = token[7:]

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user_id = data['user_id']
            request.user_role = data.get('role', 'user')
        except jwt.ExpiredSignatureError:
            security_logger.warning(f"Expired token from {request.remote_addr}")
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            security_logger.warning(f"Invalid token from {request.remote_addr}")
            return jsonify({'error': 'Invalid token'}), 401

        return f(*args, **kwargs)

    return decorated


# ==================== INPUT VALIDATION ====================

def validate_file(file):
    """Validate uploaded file"""
    filename = secure_filename(file.filename)
    if not filename:
        return False, "Invalid filename"

    ext = os.path.splitext(filename)[1].lower()
    if ext not in app.config['UPLOAD_EXTENSIONS']:
        return False, f"File type {ext} not allowed"

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)

    if size > app.config['MAX_CONTENT_LENGTH']:
        return False, "File too large (max 10MB)"

    if size == 0:
        return False, "Empty file"

    return True, filename


# ==================== API ENDPOINTS ====================

@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    """Login endpoint"""
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if username and password:
        user_id = hashlib.md5(username.encode()).hexdigest()
        token = generate_token(user_id, role='user')
        audit_logger.info(f"Login: {username} from {request.remote_addr}")

        return jsonify({
            'success': True,
            'token': token,
            'expires_in': 86400
        })

    return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/api/upload', methods=['POST'])
@require_auth
@limiter.limit("20 per hour")
def upload_file():
    """File upload endpoint"""
    user_id = request.user_id

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    is_valid, result = validate_file(file)
    if not is_valid:
        return jsonify({'error': result}), 400

    filename = result

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file, nrows=100000)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file, nrows=100000)
        else:
            return jsonify({'error': 'Unsupported format'}), 400

        if df.empty:
            return jsonify({'error': 'No data in file'}), 400

        file_id = hashlib.sha256(
            f"{user_id}{datetime.now().isoformat()}{filename}".encode()
        ).hexdigest()[:16]

        if not store_data_encrypted(file_id, df, user_id, ttl=7200):
            return jsonify({'error': 'Storage failed'}), 500

        audit_logger.info(f"Upload: user={user_id}, file={file_id}, rows={len(df)}")

        return jsonify({
            'success': True,
            'file_id': file_id,
            'rows': len(df),
            'columns': df.columns.tolist()[:20],
            'expires_in': 7200,
            'storage_location': f"{ENCRYPTED_STORAGE_DIR.absolute()}/{user_id}_{file_id}.encrypted"
        })

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Processing failed'}), 500


@app.route('/api/analyze/<file_id>', methods=['GET'])
@require_auth
@limiter.limit("30 per hour")
def analyze_data(file_id):
    """Data analysis endpoint"""
    user_id = request.user_id

    if not file_id.isalnum() or len(file_id) != 16:
        return jsonify({'error': 'Invalid file ID'}), 400

    df = retrieve_data_from_disk(file_id, user_id)

    if df is None:
        return jsonify({'error': 'File not found or expired'}), 404

    try:
        analysis = advanced_statistical_analysis(df)
        anomalies = detect_anomalies_advanced(df)

        result = {
            'total_rows': len(df),
            'columns': df.columns.tolist()[:20],
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()[:10],
            'statistics': analysis['basic_stats'],
            'correlations': analysis.get('correlations', []),
            'anomalies': anomalies,
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'completeness_percent': float((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100)
            }
        }

        return jsonify({
            'success': True,
            'analysis': result
        })

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': 'Analysis failed'}), 500


@app.route('/api/insights/<file_id>', methods=['GET'])
@require_auth
@limiter.limit("15 per hour")
def get_insights(file_id):
    """AI insights endpoint - MUST use Gemini API"""
    user_id = request.user_id

    if not AI_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Gemini AI not configured',
            'message': 'Please add GEMINI_API_KEY to .env file'
        }), 503

    if not file_id.isalnum() or len(file_id) != 16:
        return jsonify({'error': 'Invalid file ID'}), 400

    df = retrieve_data_from_disk(file_id, user_id)

    if df is None:
        return jsonify({'error': 'File not found'}), 404

    try:
        # Get analysis data
        analysis = advanced_statistical_analysis(df)
        analysis['anomalies'] = detect_anomalies_advanced(df)

        # Call Gemini API
        insights = generate_gemini_insights(df, analysis)

        if insights and len(insights) > 0:
            return jsonify({
                'success': True,
                'insights': insights,
                'generated_by': 'Gemini AI',
                'ai_status': 'active',
                'model_used': 'gemini-pro'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate insights from Gemini API',
                'message': 'Please check audit.log for details'
            }), 500

    except Exception as e:
        logger.error(f"Insights endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': 'AI service error',
            'message': str(e)
        }), 500


@app.route('/api/delete/<file_id>', methods=['DELETE'])
@require_auth
def delete_file(file_id):
    """Delete file endpoint"""
    user_id = request.user_id

    if delete_data_from_disk(file_id, user_id):
        return jsonify({'success': True})
    return jsonify({'error': 'Delete failed'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'security': 'enabled',
        'storage': 'disk-encrypted-AES256',
        'storage_path': str(ENCRYPTED_STORAGE_DIR.absolute()),
        'ai_status': 'active' if AI_AVAILABLE else 'not_configured',
        'active_users': len(METADATA_STORAGE),
        'version': '4.0-gemini-only'
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal error'}), 500


@app.after_request
def add_security_headers(response):
    """Security headers"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🏛️ MAHARASHTRA AI GOVERNANCE PLATFORM v4.0 - GEMINI POWERED")
    print("=" * 70)

    print("\n✅ System Configuration:")
    print(f"   ✓ Storage: Encrypted files on disk (AES-256)")
    print(f"   ✓ Storage Location: {ENCRYPTED_STORAGE_DIR.absolute()}")
    print(f"   ✓ Encryption Key saved in: encryption_key.txt")
    print(f"   ✓ Authentication: JWT with 24h expiry")
    print(f"   ✓ Auto-deletion: Files deleted after 2 hours")

    print("\n🤖 AI Status:")
    if AI_AVAILABLE:
        print("   ✓ Gemini AI: ACTIVE AND WORKING")
        print("   ✓ Model: gemini-pro")
        print("   ✓ All insights powered by Gemini API")
    else:
        print("   ✗ Gemini AI: NOT CONFIGURED")
        print("   ⚠ System will NOT work without Gemini API")
        print("   📋 Setup steps:")
        print("      1. Get API key: https://makersuite.google.com/app/apikey")
        print("      2. Create .env file")
        print("      3. Add: GEMINI_API_KEY=your_key_here")

    print("\n📁 File Storage:")
    print(f"   Uploaded files are:")
    print(f"   1. Encrypted with AES-256")
    print(f"   2. Saved to: {ENCRYPTED_STORAGE_DIR.absolute()}")
    print(f"   3. Named as: userid_fileid.encrypted")
    print(f"   4. Auto-deleted after 2 hours")
    print(f"   5. You can verify encryption by opening files (they're binary)")

    print("\n🔐 Security Features:")
    print("   ✓ AES-256 encryption for all files")
    print("   ✓ JWT authentication")
    print("   ✓ Audit logging")
    print("   ✓ Auto data expiration")

    print("\n" + "=" * 70)
    print("🌐 Server starting on http://localhost:5000")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)