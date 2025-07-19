from openai import OpenAI
from typing import List, Dict, Any, Callable, Optional
import json
from app.utils.fact_checker import FactChecker

class ChatAssistant:
    def __init__(self, api_key: str, anndata_model=None):
        # Store data model first so initialize() can access metadata provider
        self.anndata_model = anndata_model

        self.client = None
        if api_key != '':
            self.set_api_key(api_key)

        self.messages: list[dict] = []
        self.visible_messages: list[dict] = []
        # Build initial system prompt (may use metadata)
        self.initialize()
        self.fact_checker = FactChecker(self.anndata_model)
        if self.client:
            self.fact_checker.set_client(self.client)
        self.functions = [
            # self.get_query_data_function(),
            self.get_highlight_data_function()
        ]

    def is_ready(self):
        return self.client is not None

    def set_api_key(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        # pass client to fact checker
        if hasattr(self, 'fact_checker') and self.fact_checker:
            self.fact_checker.set_client(self.client)

    def initialize(self, system_prompt: Optional[str] = None):
        base_prompt = (
            "You are an AI assistant for analyzing single-cell RNA sequencing data. "
            "When provided with cell information, help interpret the biological significance and provide insights."
        ) if system_prompt is None else system_prompt

        # Append dataset summary if available
        provider = getattr(self.anndata_model, "metadata", None) if self.anndata_model else None
        if provider is not None:
            try:
                dataset_summary = provider.dataset_cluster_summaries()
                base_prompt += "\n\nDataset overview:\n" + dataset_summary
            except Exception as e:
                # Fail silently; still usable
                print("Warning: could not append dataset summary to system prompt:", e)

        self.messages = [{"role": "system", "content": base_prompt}]
        return True
    
    def send_message(self, text: str, cell_ids: List[str] = None,
                     history_update_handler: Callable[[str], None] = None,
                     highlight_cells: Callable[[str], None] = None,
                     query_cells: Callable[[str], None] = None):

        user_message = self.compose_query(text, cell_ids)
        
        self.messages.append({"role": "user", "content": user_message})
        self.visible_messages.append({"role": "user", "content": text})
        if history_update_handler:
            history_update_handler(self.get_visible_conversation_history())
        
        # Get response from OpenAI API
        response = self.submit_conversation()
        response_message = response.choices[0].message
        while hasattr(response_message, 'tool_calls') and response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "query_cells":
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query", "")
                    query_results = query_cells(query)
                    self.messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": "query_cells",
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        ]
                    })

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(query_results)
                    })
                    self.visible_messages.append({"role": "assistant", f"content": f"querying data: {query}"})
                    if history_update_handler:
                        history_update_handler(self.get_visible_conversation_history())
                if tool_call.function.name == "highlight_cells":
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query", "")
                    highlight_cells(query)
                    self.messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": "query_cells",
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        ]
                    })

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Calls were successfully marked on the plot. You do not need to rerun this function. Continue conversation."
                    })
                    self.visible_messages.append({"role": "assistant", f"content": f"Highlighting cells: {query}"})
                    if history_update_handler:
                        history_update_handler(self.get_visible_conversation_history())
            response = self.submit_conversation()
            response_message = response.choices[0].message
        assistant_response = response_message.content
        self.messages.append({"role": "assistant", "content": assistant_response})
        self.visible_messages.append({"role": "assistant", "content": assistant_response})
        if history_update_handler:
            history_update_handler(self.get_visible_conversation_history())

        # ---------------- Fact checking ----------------
        validation_results = self.fact_checker.validate_response(assistant_response)
        if validation_results:
            summary_lines = []
            for item in validation_results:
                outcome = item.get("result", "NOT_VERIFIABLE")
                if outcome == "TRUE":
                    status = "valid"
                elif outcome == "FALSE":
                    status = "invalid"
                else:
                    status = "not verifiable"

                line = f"{item.get('claim')}: {status}"
                # Append reason if available and not None
                if item.get("reason"):
                    line += f" (reason: {item['reason']})"
                summary_lines.append(line)
            validation_text = "Fact check:\n" + "\n".join(summary_lines)
        else:
            validation_text = "Fact check: No data-related claims detected."
        self.messages.append({"role": "assistant", "content": validation_text})
        self.visible_messages.append({"role": "assistant", "content": validation_text})
        if history_update_handler:
            history_update_handler(self.get_visible_conversation_history())
        # -------------------------------------------------
        return assistant_response

    def compose_query(self, text: str, cell_ids: List[str] = None):
        user_message = text

        if cell_ids and self.anndata_model:
            provider = getattr(self.anndata_model, "metadata", None)
            if provider is not None:
                try:
                    summary = provider.selection_summary(cell_ids)
                    user_message += f"\n\nSelected data summary:\n{summary}"
                except Exception as e:
                    user_message += f"\n\nSelected cell IDs: {', '.join(cell_ids)}\n(Error summarising selection: {str(e)})"
            else:
                user_message += f"\n\nSelected cell IDs: {', '.join(cell_ids)}"
        elif cell_ids:
            user_message += f"\n\nSelected cell IDs: {', '.join(cell_ids)}"
        return user_message

    def submit_conversation(self):
        return self.client.chat.completions.create(
            # model="gpt-4.1-mini-2025-04-14",
            model='gpt-4.1-2025-04-14',
            messages=self.messages,
            tools=self.functions,
            tool_choice="auto"
        )
    
    def get_conversation_history(self, formatted: bool = True):
        if not formatted:
            return self.messages
        
        formatted_history = []
        for message in self.messages:
            role = message["role"]
            content = message.get("content")
            
            if role == "system":
                continue
            elif role == "user":
                formatted_history.append(f"User: {content}")
            elif role == "assistant":
                if content:
                    formatted_history.append(f"Assistant: {content}")
            elif role == "tool":
                formatted_history.append(f"[Tool Response: {content}]")
        
        return "\n\n".join(formatted_history)

    def get_visible_conversation_history(self):
        formatted_history = []
        for message in self.visible_messages:
            role = message["role"]
            content = message.get("content")

            if role == "system":
                continue
            elif role == "user":
                formatted_history.append(f"User: {content}")
            elif role == "assistant":
                if content:
                    formatted_history.append(f"Assistant: {content}")
            elif role == "tool":
                formatted_history.append(f"[Tool Response: {content}]")

        return "\n\n".join(formatted_history)


    def get_query_data_function(self):
        return {
                "type": "function",
                "function": {
                    "name": "query_cells",
                    "description": "Query cells from the dataset based on similarity to the provided text. The input text has to be descriptive, and you should use this tool only to extract additional data from the dataset needed to respond to user. The response of this function will be list of metadata added to the similar cells. Use the tool at most 3 times per response.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query text to find similar cells. Be very specyfic, at least 5 sentences of description metadata of which cells you want in response"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }

    def get_highlight_data_function(self):
        return {
            "type": "function",
            "function": {
                "name": "highlight_cells",
                "description": "You can use this function to highlight some cells on the plot for the user. To do that you provide textual description of the cells you want to highlight. Then cells that are closest to your description are shown on the plot in different color. This function always returns true, execute this function always when you are discussing some group of cells or you are asked about some cells in dataset. Make sure you do not execute this function many consecutive times, after one execution just describe to user what you did.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query text to find cells to be highlighted. Be very specific, at least 5 sentences of description of desired cells"
                        }
                    },
                    "required": ["query"]
                }
            }
        }