import os
from environs import Env

env = Env()

MAX_NEW_TOKENS = env.int("MAX_NEW_TOKENS", 350)
DOCS_BASE_PATH = env.str("DOCS_BASE_PATH", "/content/drive/Shareddrives/Engineering/Chatbot/thejas/data/")
INDEX_BASE_PATH = env.str("INDEX_BASE_PATH", "/content/drive/Shareddrives/Engineering/Chatbot/thejas/indexes/")
HF_INDEX_BASE_PATH = env.str("HF_INDEX_BASE_PATH", "/content/drive/Shareddrives/Engineering/Chatbot/thejas/indexes/hf/")

DOC_INDEX_NAME_PREFIX = env.list("DOC_INDEX_NAME_PREFIX", ["doc_index"])
API_INDEX_NAME_PREFIX = env.list("API_INDEX_NAME_PREFIX", ["api_index_csv"])
DEFAULT_MODEL_NAME = env.str("DEFAULT_MODEL_NAME", "thr10/thr-wlm-15b-3gb")
USE_4_BIT_QUANTIZATION = env.bool("USE_4_BIT_QUANTIZATION", True)
SET_DEVICE_MAP = env.bool("SET_DEVICE_MAP", True)

GDRIVE_MOUNT_BASE_PATH = env.str("GDRIVE_MOUNT_BASE_PATH", "/content/drive")

OPEN_AI_TEMP = env.int("OPEN_AI_TEMP", 0)

OPEN_AI_MODEL_NAME = env.str("OPEN_AI_MODEL_NAME", "gpt-3.5-turbo")

USER_NAME = env.str("USER_NAME", "user@infoworks.io")

DEFAULT_DEVICE_MAP = env.str("DEFAULT_DEVICE_MAP", "auto")

MYSQL_HOST = env.str("MYSQL_HOST", "35.224.111.132")
MYSQL_USER = env.str("MYSQL_USER", "infoworks")
MYSQL_PASSWD = env.str("MYSQL_PASSWD", "IN11**rk")
MYSQL_DB = env.str("MYSQL_DB", "generative_ai")

DEFAULT_PROMPT_FOR_DOC = """

Use the below context and embeddings to answer the user questions

CONTEXT: 
{context}
=========

QUESTION: {question} 

"""

DEFAULT_PROMPT_FOR_API = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

You are a REST API assistant working at Infoworks, but you are also an expert programmer.
You are to complete the user request by composing a series of commands.
Use the minimum number of commands required.

The commands you have available are:
| Command | Arguments | Description | Output Format |
| --- | --- | --- | --- |
| message | message | Send the user a message | null |
| input | question | Ask the user for an input | null |
| execute | APIRequest | execute an Infoworks v3 REST API request | null |

always use the above mentioned commands and output json as shown in examples below, dont add anything extra to the answer.

Example 1:
[
  {{
    "command": "execute",
    "arguments": {{
      "type": "GET",
      "url": "http://10.37.0.7:3001/v3/sources",
      "headers": {{
        "Content-Type": "application/json",
        "Authorization": "Bearer {{refresh_token}}"
      }},
      "body": ""
    }}
  }}
]

Example 2:
Request: List all teradata sources
Response:
[
  {{
    "command": "execute",
    "arguments": {{
      "type": "GET",
      "url": "http://10.37.0.7:3001/v3/sources",
      "headers": {{
        "Content-Type": "application/json",
        "Authorization": "Bearer {{refresh_token}}"
      }},
      "body": ""
    }}
  }}
]

Example 3:
User Request: create a teradata source with source name \"Teradata_sales\"
Response:
[
  {{
    "command": "input",
    "arguments" : "Enter the source name"
  }},
  {{
    "command": "input",
    "arguments" : "Enter the source type"
  }},
  {{
    "command": "input",
    "arguments" : "Enter the source sub type"
  }},
  {{
    "command": "input",
    "arguments" : "Enter the data lake path"
  }},
  {{
    "command": "input",
    "arguments" : "Enter the environment id"
  }},
  {{
    "command": "input",
    "arguments" : "Enter the storage id"
  }},
  {{
    "command": "input",
    "arguments" : "Enter the data lake schema"
  }},
  {{
    "command": "input",
    "arguments" : "Enter the is_oem_connector"
  }},
  {{
    "command": "execute",
    "arguments": {{
      "type": "POST"
      "url": "http://10.37.0.7:3001/api/v3/sources",
      "headers": "{{\"Content-Type\": \"application/json\", \"Authorization\": \"Bearer {{refresh_token}}\"}}",
      "body": {{
        "name": "{{{{input_0}}}}",
        "environment_id": "{{{{input_4}}}}",
        "storage_id": "{{{{input_5}}",
        "data_lake_schema": "{{{{input_6}}}}"
        "data_lake_path": "{{{{input_3}}}}",
        "type": "{{{{input_1}}}}",
        "sub_type": "{{{{input_2}}}}",
        "is_oem_connector": "{{{{input_7}}}}"
      }}
    }}
  }}
]

{context}

IMPORTANT - Output the commands in JSON as an abstract syntax tree. Do not respond with any text that isn't part of a command.
IMPORTANT - Do not explain yourself or Do not give Any Explanation
IMPORTANT - You are an expert at generating commands and You can only generate commands.
IMPORTANT - Do not assume any values of put or post or patch requests. always get the input from user any params.
IMPORTANT - Infoworks instance ip is 10.37.0.7 and port 3001
IMPORTANT - Authenticate all execute commands using refresh_token assume user already has that information

Understand the following request and generate the minimum set of commands as shown in examples above to complete it.

Question: {question}
"""

API_QUESTION_PROMPT = env.str("API_QUESTION_PROMPT", DEFAULT_PROMPT_FOR_API)
DOC_QUESTION_PROMPT = env.str("DOC_QUESTION_PROMPT", DEFAULT_PROMPT_FOR_DOC)

DEFAULT_CSV_PARSE_ARGS = {
        "delimiter": ",",
        "fieldnames": ["Method", "Path", "Operation", "Description", "Query Parameters", "Request Parameters"],
    }

CSV_DOC_PARSE_ARGS = env.dict("CSV_DOC_PARSE_ARGS", DEFAULT_CSV_PARSE_ARGS)

CSV_DOC_EMBEDDING_SOURCE_COLUMN = env.str("CSV_DOC_EMBEDDING_SOURCE_COLUMN", "Description")