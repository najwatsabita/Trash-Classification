# upload_to_drive.py
import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload_to_drive():
    # Load credentials from environment variable
    creds_json = os.environ.get('GOOGLE_DRIVE_CREDENTIALS')
    folder_id = os.environ.get('GOOGLE_DRIVE_MODEL_FOLDER_ID')
    
    creds_dict = json.loads(creds_json)
    credentials = Credentials.from_authorized_user_info(creds_dict)

    # Create Drive API service
    service = build('drive', 'v3', credentials=credentials)
    
    # File metadata
    file_metadata = {
        'name': 'Trash-Classification.ckpt',
        'parents': [folder_id]
    }
    
    # Upload file
    media = MediaFileUpload('Trash-Classification.ckpt',
                           mimetype='application/octet-stream',
                           resumable=True)
    
    file = service.files().create(body=file_metadata,
                                 media_body=media,
                                 fields='id').execute()
    
    print(f'File ID: {file.get("id")}')

if __name__ == '__main__':
    upload_to_drive()