# -*- coding: utf-8 -*-
"""
RAG-система для консультаций по ювелирным изделиям (ЛОКАЛЬНАЯ ВЕРСИЯ)
Использует FAISS для векторного хранилища и локальную модель Llama для генерации ответов
Эта версия НЕ требует OpenAI API и работает полностью локально
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
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Импорт эмбеддингов
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)

# Импорты для локальной модели
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class RAGSystemLocal:
    """Класс для работы с RAG-системой (локальная версия)"""
    
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
        self.rag_chain = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
            print("⚠ Нет документов. Создается база с информацией по умолчанию.")
            documents = [
                Document(page_content="Золото - благородный металл, требующий бережного ухода.", metadata={}),
                Document(page_content="Серебро темнеет со временем, его нужно регулярно чистить.", metadata={}),
                Document(page_content="Бриллианты - самые твердые драгоценные камни.", metadata={})
            ]
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
                "name": "Кольцо с рубином",
                "description": "Роскошное золотое кольцо с крупным натуральным рубином.",
                "usage": "Идеально для особых случаев и торжественных мероприятий.",
                "price": "35000 руб.",
                "url": "https://example.com/product/11"
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
        """Инициализация локальной языковой модели"""
        print(f"Инициализация локальной модели (устройство: {self.device})...")
        print("⚠ ВНИМАНИЕ: Загрузка модели может занять несколько минут...")
        
        model_name = "IlyaGusev/saiga_llama3_8b"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if self.device == "cuda":
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
                model.to(self.device)
            
            model.eval()
            
            terminators = [tokenizer.eos_token_id]
            if "<|eot_id|>" in tokenizer.get_vocab():
                terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
            
            text_generation_pipeline = pipeline(
                model=model,
                tokenizer=tokenizer,
                task="text-generation",
                temperature=0.5,
                do_sample=True,
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens=500,
                eos_token_id=terminators,
            )
            
            self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
            print("✓ Локальная модель Llama загружена и готова")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("⚠ Используется упрощенный режим (без генерации)")
            self.llm = None
    
    def create_rag_chain(self):
        """Создание RAG цепочки"""
        if self.llm is None:
            print("⚠ LLM не инициализирована, RAG цепочка не создана")
            return
        
        prompt_template = """
<|start_header_id|>user<|end_header_id|>
Ты — умный ассистент, специализирующийся на ювелирных украшениях.

Контекст из базы знаний:
{context}

Доступные товары:
{products}

Правила:
- Отвечай на русском языке
- Будь полезным и дружелюбным
- Если не знаешь ответа, скажи об этом
- При предложении товаров будь конкретным

Вопрос: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        prompt = PromptTemplate(
            input_variables=["context", "products", "question"],
            template=prompt_template
        )
        
        def format_docs(docs):
            if not docs:
                return "Информация не найдена"
            return "\n\n".join(doc.page_content for doc in docs)
        
        def format_products(products):
            if not products:
                return "Товары не найдены"
            return "\n".join(
                f"• {doc.metadata['name']}: {doc.page_content} (Цена: {doc.metadata['price']})"
                for doc in products
            )
        
        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "products": self.catalog_retriever | format_products,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("✓ RAG цепочка создана")
    
    def query(self, question: str) -> str:
        """Выполнение запроса к RAG-системе"""
        if self.rag_chain is None:
            # Упрощенный режим без LLM
            try:
                products_docs = self.catalog_retriever.invoke(question)
                if products_docs:
                    result = "Вот что я нашел в каталоге:\n\n"
                    for doc in products_docs:
                        result += f"• {doc.metadata['name']}\n"
                        result += f"  {doc.page_content}\n"
                        result += f"  Цена: {doc.metadata['price']}\n"
                        result += f"  Ссылка: {doc.metadata['url']}\n\n"
                    return result
                else:
                    return "К сожалению, не нашел подходящих товаров."
            except Exception as e:
                return f"Ошибка поиска: {e}"
        
        try:
            result = self.rag_chain.invoke(question)
            return result
        except Exception as e:
            return f"Ошибка при генерации ответа: {e}"
    
    def run_interactive(self):
        """Запуск интерактивного режима"""
        print("\n" + "="*60)
        print("RAG-СИСТЕМА: КОНСУЛЬТАНТ ПО ЮВЕЛИРНЫМ ИЗДЕЛИЯМ (ЛОКАЛЬНО)")
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
    print("ИНИЦИАЛИЗАЦИЯ RAG-СИСТЕМЫ (ЛОКАЛЬНАЯ ВЕРСИЯ)")
    print("="*60)
    
    # Определение базовой директории
    base_dir = Path(__file__).parent
    
    try:
        # Создание экземпляра RAG-системы
        rag = RAGSystemLocal(base_dir)
        
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

