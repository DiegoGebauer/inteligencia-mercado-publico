"""Pruebas del diagnóstico HTTP con respuestas simuladas; no usa el ticket real."""
from datetime import datetime, timezone, timedelta
from email.utils import format_datetime
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
path = Path(__file__).resolve().parents[1] / 'src/api/mercado_publico_client.py'
spec = importlib.util.spec_from_file_location('client_retry_under_test', path)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class RetryMetadataTests(unittest.TestCase):
    def test_seconds_date_and_invalid_retry_after(self):
        self.assertEqual(client._retry_after_seconds('17'), 17)
        self.assertEqual(client._retry_after_seconds('0'), 0)
        for value in [None, '', '-1', '1.5', 'https://sitio/?ticket=secreto']:
            self.assertIsNone(client._retry_after_seconds(value))
        now = datetime.now(timezone.utc).replace(microsecond=0)
        future = format_datetime(now + timedelta(seconds=40), usegmt=True)
        self.assertTrue(39 <= client._retry_after_seconds(future) <= 40)
        self.assertEqual(client._retry_after_seconds(format_datetime(now - timedelta(seconds=40), usegmt=True)), 0)

    def test_http_error_retains_safe_metadata_and_old_runtime_interface(self):
        response = Mock(status_code=429, headers={'Retry-After': '17'})
        response.raise_for_status.side_effect = client.requests.HTTPError('https://sitio/?ticket=secreto', response=response)
        with patch.object(client, '_load_ticket', return_value='ticket-de-prueba'), patch.object(client.requests, 'get', return_value=response):
            with self.assertRaises(client.MercadoPublicoAPIError) as raised:
                client.get_licitacion_por_codigo('1-1-LE26')
        error = raised.exception
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.retry_after_seconds, 17)
        self.assertNotIn('secreto', str(error))
        self.assertNotIn('ticket-de-prueba', str(error))

    def test_connection_failure_does_not_disclose_url(self):
        with patch.object(client, '_load_ticket', return_value='ticket-de-prueba'), patch.object(client.requests, 'get', side_effect=client.requests.ConnectionError('https://sitio/?ticket=secreto')):
            with self.assertRaises(client.MercadoPublicoAPIError) as raised:
                client.get_licitaciones_por_fecha('15102026')
        self.assertIsNone(raised.exception.status_code)
        self.assertNotIn('secreto', str(raised.exception))


if __name__ == '__main__':
    unittest.main()
