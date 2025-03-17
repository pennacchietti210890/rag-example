import pytest
import base64
from unittest.mock import patch

from backend.main import xor_encrypt_decrypt, decrypt_api_key


@pytest.mark.unit
class TestEncryption:
    
    def test_xor_encrypt_decrypt_roundtrip(self):
        """Test that encryption and decryption work correctly in a roundtrip"""
        # Given
        original_data = "gsk_test_api_key_12345"
        encryption_key = "test_encryption_key"
        
        # When
        encrypted = xor_encrypt_decrypt(original_data, encryption_key)
        decrypted = decrypt_api_key(encrypted, encryption_key)

        # Then
        assert encrypted != original_data  # Encryption changed the data
        assert decrypted == original_data  # Decryption restored the original data
    
    def test_xor_encrypt_decrypt_different_keys(self):
        """Test that different encryption keys produce different results"""
        # Given
        original_data = "gsk_test_api_key_12345"
        key1 = "encryption_key_1"
        key2 = "encryption_key_2"
        
        # When
        encrypted1 = xor_encrypt_decrypt(original_data, key1)
        encrypted2 = xor_encrypt_decrypt(original_data, key2)
        
        # Then
        assert encrypted1 != encrypted2
    
    def test_decrypt_api_key(self):
        """Test the dedicated decrypt_api_key function"""
        # Given
        original_data = "gsk_test_api_key_12345"
        encryption_key = "test_encryption_key"
        encrypted = xor_encrypt_decrypt(original_data, encryption_key)
        
        # When
        decrypted = decrypt_api_key(encrypted, encryption_key)
        
        # Then
        assert decrypted == original_data
    
    def test_decrypt_api_key_error_handling(self):
        """Test error handling in decrypt_api_key"""
        # Given
        invalid_encrypted = "not_valid_base64"
        encryption_key = "test_encryption_key"
        
        # When
        with patch('backend.main.logger') as mock_logger:
            result = decrypt_api_key(invalid_encrypted, encryption_key)
        
        # Then
        assert result == "invalid_key"  # Returns fallback value
        mock_logger.error.assert_called_once()  # Error was logged 