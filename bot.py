from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.graph import MessagesState, StateGraph ,START ,END
from typing import Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages, AnyMessage


load_dotenv()

groq_key = os.getenv("GROQ_KEY")
tavily_key = os.getenv("TAVILY_KEY")
os.environ['TAVILY_API_KEY'] =  tavily_key

class chatbot:
    def __init__(self):
        self.llm = ChatGroq(model="gemma2-9b-it",  api_key=groq_key)
    
    def call_tool(self):
        tool = TavilySearchResults(max_results=5)
        tools = [tool]
        self.tool_node = ToolNode(tools)
        self.llm_with_tool = self.llm.bind_tools(tools)


    def call_model(self, state:MessagesState):
        messages = state['messages']
        response = self.llm_with_tool.invoke(messages)

        return {"messages": [response]}
    
    def router_function(self,state: MessagesState) -> Literal["tools",END]:
        messages = state["messages"]
        last_messages =  messages[-1]
        if last_messages.tool_calls:
            return "tools"
        
        return  END

    def __call__(self):
        self.call_tool()
        workflow = StateGraph(MessagesState)
        workflow.add_node('agent', self.call_model)
        workflow.add_node('tools',self.tool_node)
        workflow.add_edge(START,  'agent')
        workflow.add_conditional_edges('agent',self.router_function,{'tools':'tools',END:END})
        workflow.add_edge('tools','agent')
        self.app = workflow.compile()
        return self.app
    

if  __name__ == "__main__":
    chatbot = chatbot()
    app = chatbot()
    response = app.invoke({'messages':['who is superman']})
    print(response['messages'][-1].content)  # prints the response from the chatbot
