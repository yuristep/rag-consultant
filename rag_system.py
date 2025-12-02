# -*- coding: utf-8 -*-
"""
RAG-система для консультаций по ювелирным изделиям
Использует FAISS для векторного хранилища и OpenAI GPT-4o-mini для генерации ответов
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Настройка кодировки для Windows
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'

# Импорты LangChain и зависимостей
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Импорт эмбеддингов с обработкой устаревшей версии
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)

# API ключи (ВАЖНО: используйте .env файл для хранения ключей!)
# Скопируйте env.example в .env и укажите свои ключи
import os
from dotenv import load_dotenv

load_dotenv()  # Загрузка переменных из .env файла

# Если .env не используется, укажите ключи здесь (НЕ рекомендуется для продакшена)
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"
    
if not os.getenv("UNSTRUCTURED_API_KEY"):
    os.environ["UNSTRUCTURED_API_KEY"] = "your-unstructured-api-key-here"
    os.environ["UNSTRUCTURED_URL"] = "https://api.unstructuredapp.io/general/v0/general"


class RAGSystem:
    """Класс для работы с RAG-системой"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.output_dir = base_dir / "output"
        self.faiss_index_dir = base_dir / "faiss_index"
        self.catalog_file = base_dir / "catalog.json"
        
        # Создание необходимых директорий
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.faiss_index_dir.mkdir(exist_ok=True)
        
        # Инициализация компонентов
        self.embeddings_model = None
        self.db = None
        self.retriever = None
        self.catalog_db = None
        self.catalog_retriever = None
        self.llm = None
        self.custom_rag_chain = None
        
    def initialize_embeddings(self):
        """Инициализация модели эмбеддингов"""
        print("Загрузка модели эмбеддингов...")
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cpu'}
        )
        print("✓ Модель эмбеддингов загружена")
        
    def load_documents_from_output(self) -> list:
        """Загрузка обработанных документов из директории output"""
        documents = []
        
        try:
            from unstructured.staging.base import elements_from_json
            
            if not self.output_dir.exists() or not any(self.output_dir.iterdir()):
                print("⚠ Директория output пуста. Создается база с пустым документом.")
                return []
            
            for file_path in self.output_dir.iterdir():
                if file_path.suffix == '.json':
                    try:
                        file_elements = elements_from_json(filename=str(file_path))
                        for element in file_elements:
                            if hasattr(element, "text") and element.text:
                                metadata = element.metadata.to_dict() if hasattr(element.metadata, 'to_dict') else {}
                                documents.append(Document(page_content=element.text, metadata=metadata))
                    except Exception as e:
                        print(f"⚠ Ошибка при чтении файла {file_path.name}: {e}")
                        
        except ImportError:
            print("⚠ unstructured.staging.base недоступен. Пропуск загрузки документов.")
            
        return documents
    
    def create_knowledge_base(self):
        """Создание базы знаний из документов"""
        print("Создание базы знаний...")
        
        documents = self.load_documents_from_output()
        
        if len(documents) == 0:
            print("⚠ Нет документов. Создается база с временным документом.")
            documents = [Document(page_content="Информация о ювелирных изделиях", metadata={})]
        else:
            print(f"✓ Загружено документов: {len(documents)}")
        
        # Создание FAISS базы данных
        self.db = FAISS.from_documents(documents, self.embeddings_model)
        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # Сохранение базы данных
        save_path = str(self.faiss_index_dir.resolve())
        self.db.save_local(save_path)
        print(f"✓ База знаний сохранена в {save_path}")
        
    def load_knowledge_base(self):
        """Загрузка существующей базы знаний"""
        save_path = str(self.faiss_index_dir.resolve())
        
        if not (self.faiss_index_dir / "index.faiss").exists():
            print("⚠ Сохраненная база не найдена. Создание новой...")
            self.create_knowledge_base()
        else:
            print("Загрузка существующей базы знаний...")
            self.db = FAISS.load_local(save_path, self.embeddings_model, allow_dangerous_deserialization=True)
            self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            print("✓ База знаний загружена")
    
    def create_catalog(self):
        """Создание каталога товаров"""
        catalog = [
            {
                "name": "Кольцо с бриллиантом",
                "description": "Элегантное кольцо с высококачественным бриллиантом 0.5 карата.",
                "usage": "Идеально подходит для особых случаев, таких как помолвка или свадьба.",
                "price": "15000 руб.",
                "url": "https://example.com/product/1"
            },
            {
                "name": "Серебряные серьги с аметистами",
                "description": "Серьги из 925 стерлингового серебра с аметистами высшей пробы.",
                "usage": "Подходят для ежедневного ношения или торжественных мероприятий.",
                "price": "8000 руб.",
                "url": "https://example.com/product/2"
            },
            {
                "name": "Золотая подвеска с изумрудом",
                "description": "Изысканная подвеска из 14-каратного золота с натуральным изумрудом.",
                "usage": "Отлично смотрится как для повседневного, так и для вечернего образа.",
                "price": "22000 руб.",
                "url": "https://example.com/product/3"
            },
            {
                "name": "Браслет с цирконами",
                "description": "Тонкий браслет с вставками из искусственных цирконов.",
                "usage": "Подходит для праздничных случаев или в качестве стильного аксессуара.",
                "price": "3500 руб.",
                "url": "https://example.com/product/4"
            },
            {
                "name": "Серьги с жемчугом",
                "description": "Классические серьги с натуральным жемчугом и золотыми вставками.",
                "usage": "Идеально для повседневного ношения или для элегантных мероприятий.",
                "price": "10000 руб.",
                "url": "https://example.com/product/5"
            },
            {
                "name": "Часы с бриллиантами",
                "description": "Шикарные часы с инкрустацией из бриллиантов на циферблате.",
                "usage": "Для людей, ценящих изысканный стиль и точность.",
                "price": "50000 руб.",
                "url": "https://example.com/product/6"
            },
            {
                "name": "Кулон с топазом",
                "description": "Изящный кулон с натуральным топазом, украшенный серебряной оправой.",
                "usage": "Можно носить как на праздники, так и для ежедневного стиля.",
                "price": "7000 руб.",
                "url": "https://example.com/product/7"
            },
            {
                "name": "Золотые обручальные кольца",
                "description": "Пара золотых обручальных колец с матовой отделкой и гравировкой.",
                "usage": "Идеально для жениха и невесты в день свадьбы.",
                "price": "25000 руб.",
                "url": "https://example.com/product/8"
            },
            {
                "name": "Печатка с ониксом",
                "description": "Мужская печатка с ониксом и серебряной оправой.",
                "usage": "Для стильных мужчин, предпочитающих выразительные аксессуары.",
                "price": "12000 руб.",
                "url": "https://example.com/product/9"
            },
            {
                "name": "Колье с сапфирами",
                "description": "Роскошное колье с натуральными сапфирами и бриллиантами.",
                "usage": "Для особых случаев, таких как вечеринки или торжественные мероприятия.",
                "price": "45000 руб.",
                "url": "https://example.com/product/10"
            }
        ]
        
        # Сохранение каталога в JSON
        with open(self.catalog_file, 'w', encoding='utf-8') as f:
            json.dump(catalog, f, indent=4, ensure_ascii=False)
        
        return catalog
    
    def create_catalog_database(self):
        """Создание векторной базы данных каталога"""
        print("Создание базы данных каталога...")
        
        # Загрузка или создание каталога
        if self.catalog_file.exists():
            with open(self.catalog_file, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
        else:
            catalog = self.create_catalog()
        
        # Создание документов из каталога
        documents = []
        for item in catalog:
            documents.append(Document(
                page_content=item['description'],
                metadata=item
            ))
        
        # Создание FAISS базы для каталога
        self.catalog_db = FAISS.from_documents(documents, self.embeddings_model)
        self.catalog_retriever = self.catalog_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        print(f"✓ База каталога создана ({len(documents)} товаров)")
    
    def initialize_llm(self):
        """Инициализация языковой модели"""
        print("Инициализация языковой модели...")
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
        print("✓ Модель GPT-4o-mini готова")
    
    def create_rag_chain(self):
        """Создание RAG цепочки"""
        
        prompt_template = """
Ты — умный ассистент, специализирующийся на ювелирных украшениях.

Твои основные задачи:
1. Отвечать на вопросы о ювелирных украшениях, их характеристиках и ценах
2. Помогать клиентам в выборе подходящих товаров

Контекст из базы знаний:
{context}

Доступные товары:
{products}

Правила:
- Предоставляй полезные, понятные и дружелюбные ответы
- Если не знаешь ответа, скажи: «Я не знаю»
- Не придумывай информацию
- При предложении товаров будь конкретным
- Если нужна дополнительная информация, задавай уточняющие вопросы

Вопрос: {question}
"""
        
        prompt = PromptTemplate(
            input_variables=["context", "products", "question"],
            template=prompt_template
        )
        
        class CustomRAGChain:
            def __init__(self, llm, prompt):
                self.llm = llm
                self.prompt = prompt
            
            def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                context = inputs.get("context", "")
                products = inputs.get("products", "")
                question = inputs.get("question", "")
                
                full_prompt = self.prompt.format(
                    context=context,
                    products=products,
                    question=question
                )
                
                messages = [HumanMessage(content=full_prompt)]
                response = self.llm.invoke(messages)
                
                if hasattr(response, 'content'):
                    return {"text": response.content}
                else:
                    return {"text": str(response)}
        
        self.custom_rag_chain = CustomRAGChain(self.llm, prompt)
    
    @staticmethod
    def format_docs(docs):
        """Форматирование документов для промпта"""
        if not docs:
            return "Информация не найдена"
        return "\n\n".join(doc.page_content for doc in docs)
    
    @staticmethod
    def format_products(products):
        """Форматирование товаров для промпта"""
        if not products:
            return "Товары не найдены"
        return "\n".join(
            f"• {doc.metadata['name']}\n"
            f"  Описание: {doc.page_content}\n"
            f"  Цена: {doc.metadata['price']}\n"
            f"  Ссылка: {doc.metadata['url']}\n"
            for doc in products
        )
    
    def query(self, question: str) -> str:
        """Выполнение запроса к RAG-системе"""
        try:
            # Получение контекста из базы знаний (используем invoke вместо get_relevant_documents)
            context_docs = self.retriever.invoke(question)
            context = self.format_docs(context_docs)
        except Exception as e:
            print(f"⚠ Ошибка при получении контекста: {e}")
            context = ""
        
        try:
            # Получение релевантных товаров (используем invoke вместо get_relevant_documents)
            products_docs = self.catalog_retriever.invoke(question)
            products = self.format_products(products_docs)
        except Exception as e:
            print(f"⚠ Ошибка при получении товаров: {e}")
            products = ""
        
        try:
            # Генерация ответа
            result = self.custom_rag_chain.invoke({
                "context": context,
                "products": products,
                "question": question
            })
            return result['text']
        except Exception as e:
            # Обработка ошибки OpenAI API (403 - регион не поддерживается)
            error_msg = str(e)
            if "403" in error_msg and "unsupported_country_region_territory" in error_msg:
                return "⚠ Ошибка OpenAI API: Ваш регион не поддерживается. Решения:\n1. Используйте VPN\n2. Замените API ключ на ключ из поддерживаемого региона\n3. Используйте локальную модель (см. старый код с Llama)"
            return f"Ошибка при генерации ответа: {e}"
    
    def run_interactive(self):
        """Запуск интерактивного режима"""
        print("\n" + "="*60)
        print("RAG-СИСТЕМА: КОНСУЛЬТАНТ ПО ЮВЕЛИРНЫМ ИЗДЕЛИЯМ")
        print("="*60)
        print("\nДобро пожаловать! Я помогу вам с выбором ювелирных изделий.")
        print("Введите 'exit', 'выход' или 'quit' для завершения\n")
        
        while True:
            try:
                question = input("Вы: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ["exit", "выход", "quit"]:
                    print("\nДо свидания! Буду рад помочь вам снова.")
                    break
                
                print("\nАссистент: ", end="", flush=True)
                answer = self.query(question)
                print(answer + "\n")
                
            except KeyboardInterrupt:
                print("\n\nПрограмма прервана пользователем.")
                break
            except Exception as e:
                print(f"\n⚠ Произошла ошибка: {e}\n")


def main():
    """Главная функция"""
    print("="*60)
    print("ИНИЦИАЛИЗАЦИЯ RAG-СИСТЕМЫ")
    print("="*60)
    
    # Определение базовой директории
    base_dir = Path(__file__).parent
    
    try:
        # Создание экземпляра RAG-системы
        rag = RAGSystem(base_dir)
        
        # Инициализация компонентов
        rag.initialize_embeddings()
        rag.load_knowledge_base()
        rag.create_catalog_database()
        rag.initialize_llm()
        rag.create_rag_chain()
        
        print("\n✓ Система готова к работе!\n")
        
        # Запуск интерактивного режима
        rag.run_interactive()
        
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем.")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

