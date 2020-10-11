import base64
import binascii

# message = "Python is fun"
# message_bytes = message.encode('ascii')
# base64_bytes = base64.b64encode(message_bytes)
# base64_message = base64_bytes.decode('ascii')

# print(base64_message)


# base64_message = 'UHl0aG9uIGlzIGZ1bg=='
# base64_bytes = base64_message.encode('ascii')
# message_bytes = base64.b64decode(base64_bytes)
# message = message_bytes.decode('ascii')

# print(message)


# with open("test.data",'rb') as f:
#     contents = f.read()
# print(contents)
# message_bytes = message.encode('ascii')
# encodedZip = base64.b64encode(message_bytes).decode('ascii')
# print(encodedZip)
# base64_bytes = encodedZip.encode('ascii')
# decode = base64.b64decode(base64_bytes).decode('ascii')
# print(decode)
string = "Hello World"

binascii.a2b_uu(string)
