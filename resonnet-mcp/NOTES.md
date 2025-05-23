## Claude.ai conversations.json Export Format

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "account": {
        "type": "object",
        "properties": {
          "uuid": {
            "type": "string"
          }
        }
      },
      "chat_messages": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "attachments": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "extracted_content": {
                    "type": "string"
                  },
                  "file_name": {
                    "type": "string"
                  },
                  "file_size": {
                    "type": "number"
                  },
                  "file_type": {
                    "type": "string"
                  }
                }
              }
            },
            "content": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "text": {
                    "type": "string"
                  },
                  "type": {
                    "type": "string"
                  }
                }
              }
            },
            "created_at": {
              "type": "string"
            },
            "files": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "file_name": {
                    "type": "string"
                  }
                }
              }
            },
            "sender": {
              "type": "string"
            },
            "text": {
              "type": "string"
            },
            "updated_at": {
              "type": "string"
            },
            "uuid": {
              "type": "string"
            }
          }
        }
      },
      "created_at": {
        "type": "string"
      },
      "name": {
        "type": "string"
      },
      "updated_at": {
        "type": "string"
      },
      "uuid": {
        "type": "string"
      }
    }
  }
}
```