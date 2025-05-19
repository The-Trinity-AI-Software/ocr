# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:20:37 2025

@author: HP
"""

from azure.storage.blob import BlobServiceClient
import json

def upload_result_to_blob(data):
    conn_str = "AZURE_STORAGE_CONNECTION_STRING"
    container = "output-results"
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    blob = blob_service.get_blob_client(container=container, blob="result.json")
    blob.upload_blob(json.dumps(data), overwrite=True)
