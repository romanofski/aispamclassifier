import io

from aispamclassifier import spamcheck

def test_parses_email_successfully(mocker):
    classify_mock = mocker.patch.object(spamcheck, 'classify')
    classify_mock.return_value = 'Spam'

    email = b"""This \xc4 is a mail wwith umlauts"""
    buffer = io.BytesIO(email)

    outbuffer = io.BytesIO()

    spamcheck.handle_email(buffer, 'tag', outbuffer)

    outbuffer.seek(0)
    assert outbuffer.read() == b'X-AI-Spam: Spam\n\n' + email
