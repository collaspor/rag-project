from server.db.models.knowledge_base_model import KnowledgeBaseModel
from server.db.models.knowledge_file_model import KnowledgeFileModel, FileDocModel
from server.knowledge_base.utils import KnowledgeFile
from typing import List, Dict

from server.db.session import with_async_session, async_session_scope


from sqlalchemy.future import select
from sqlalchemy import func, delete

@with_async_session
async def list_file_num_docs_id_by_kb_name_and_file_name(session,
                                                   kb_name: str,
                                                   file_name: str,
                                                   ) -> List[int]:
    '''
    列出某知识库某文件对应的所有Document的id。
    返回形式：[str, ...]
    '''
    stmt = select(FileDocModel.doc_id).filter_by(kb_name=kb_name, file_name=file_name)
    result = await session.execute(stmt)
    return [int(row[0]) for row in result.all()]


@with_async_session
async def list_docs_from_db(session,
                      kb_name: str,
                      file_name: str = None,
                      metadata: Dict = {},
                      ) -> List[Dict]:
    '''
    列出某知识库某文件对应的所有Document。
    返回形式：[{"id": str, "metadata": dict}, ...]
    '''
    stmt = select(FileDocModel).where(FileDocModel.kb_name.ilike(kb_name))
    if file_name:
        stmt = stmt.where(FileDocModel.file_name.ilike(file_name))
    for k, v in metadata.items():
        stmt = stmt.where(FileDocModel.meta_data[k].as_string() == str(v))

    result = await session.execute(stmt)
    return [{"id": x.doc_id, "metadata": x.meta_data} for x in result.scalars().all()]


@with_async_session
async def delete_docs_from_db(session,
                        kb_name: str,
                        file_name: str = None,
                        ) -> List[Dict]:
    '''
    删除某知识库某文件对应的所有Document，并返回被删除的Document。
    返回形式：[{"id": str, "metadata": dict}, ...]
    '''
    # 先查询将要被删除的文档（与本次删除处于同一会话/事务）
    select_stmt = select(FileDocModel).where(FileDocModel.kb_name.ilike(kb_name))
    if file_name:
        select_stmt = select_stmt.where(FileDocModel.file_name.ilike(file_name))
    result = await session.execute(select_stmt)
    docs = [{"id": x.doc_id, "metadata": x.meta_data} for x in result.scalars().all()]

    # 执行删除
    del_stmt = delete(FileDocModel).where(FileDocModel.kb_name.ilike(kb_name))
    if file_name:
        del_stmt = del_stmt.where(FileDocModel.file_name.ilike(file_name))
    await session.execute(del_stmt)
    await session.commit()
    return docs


@with_async_session
async def add_docs_to_db(session,
                   kb_name: str,
                   file_name: str,
                   doc_infos: List[Dict]):
    '''
    将某知识库某文件对应的所有Document信息添加到数据库。
    doc_infos形式：[{"id": str, "metadata": dict}, ...]
    '''
    # ! 这里会出现doc_infos为None的情况，需要进一步排查
    if doc_infos is None:
        print("输入的server.db.repository.knowledge_file_repository.add_docs_to_db的doc_infos参数为None")
        return False
    try:
        for doc_info in doc_infos:
            obj = FileDocModel(
                kb_name=kb_name,
                file_name=file_name,
                doc_id=doc_info['id'],
                meta_data=doc_info['metadata'],
            )
            session.add(obj)
        await session.commit()
        print("文档信息成功添加到数据库")
        return True
    except Exception as e:
        print(f"在添加文档信息时发生错误: {e}")
        await session.rollback()
        return False


@with_async_session
async def count_files_from_db(session, kb_name: str) -> int:
    stmt = (
        select(func.count())
        .select_from(KnowledgeFileModel)
        .where(KnowledgeFileModel.kb_name.ilike(kb_name))
    )
    result = await session.execute(stmt)
    return result.scalar() or 0


@with_async_session
async def list_files_from_db(session, kb_name):
    stmt = select(KnowledgeFileModel).where(KnowledgeFileModel.kb_name.ilike(kb_name))
    result = await session.execute(stmt)
    docs = [f.file_name for f in result.scalars().all()]
    return docs


@with_async_session
async def add_file_to_db(session,
                         kb_file: KnowledgeFile,
                         docs_count: int = 0,
                         custom_docs: bool = False,
                         doc_infos: List[Dict] = [],  # 形式：[{"id": str, "metadata": dict}, ...]
                         ):
    """
    将文件添加到数据库中。如果文件已经存在，则更新文件信息和版本号。

    参数：
        session: 数据库会话对象。
        kb_file: 知识文件对象，包含文件的相关信息。
        docs_count: 文档数量。
        custom_docs: 是否为自定义文档。
        doc_infos: 文档信息列表，形式为：[{"id": str, "metadata": dict}, ...]

    返回：
        bool: 如果操作成功，返回True。
    """

    stmt = select(KnowledgeBaseModel).where(KnowledgeBaseModel.kb_name == kb_file.kb_name)
    kb_result = await session.execute(stmt)
    kb = kb_result.scalars().first()

    if kb:
        stmt = select(KnowledgeFileModel).where(
            KnowledgeFileModel.kb_name.ilike(kb_file.kb_name),
            KnowledgeFileModel.file_name.ilike(kb_file.filename)
        )
        file_result = await session.execute(stmt)
        existing_file = file_result.scalars().first()

        mtime = kb_file.get_mtime()
        size = kb_file.get_size()

        if existing_file:
            existing_file.file_mtime = mtime
            existing_file.file_size = size
            existing_file.docs_count = docs_count
            existing_file.custom_docs = custom_docs
            existing_file.file_version += 1
        else:
            new_file = KnowledgeFileModel(
                file_name=kb_file.filename,
                file_ext=kb_file.ext,
                kb_name=kb_file.kb_name,
                document_loader_name=kb_file.document_loader_name,
                text_splitter_name=kb_file.text_splitter_name or "SpacyTextSplitter",
                file_mtime=mtime,
                file_size=size,
                docs_count=docs_count,
                custom_docs=custom_docs,
            )
            session.add(new_file)
            kb.file_count += 1

        # 在同一会话内写入文档信息，保证与文件记录处于同一事务
        if doc_infos:
            for doc_info in doc_infos:
                session.add(FileDocModel(
                    kb_name=kb_file.kb_name,
                    file_name=kb_file.filename,
                    doc_id=doc_info['id'],
                    meta_data=doc_info['metadata'],
                ))

        try:
            await session.commit()
        except Exception as e:
            print(f"Error committing changes: {e}")
            await session.rollback()
            raise
    else:
        print("KnowledgeBase 不存在，无法添加文件")
    return True


@with_async_session
async def delete_file_from_db(session, kb_file: KnowledgeFile):
    # 使用异步查询获取文件
    result = await session.execute(
        select(KnowledgeFileModel)
        .filter(KnowledgeFileModel.file_name.ilike(kb_file.filename),
                KnowledgeFileModel.kb_name.ilike(kb_file.kb_name))
    )
    existing_file = result.scalars().first()

    if existing_file:
        # 删除文件记录
        await session.delete(existing_file)

        # 删除该文件关联的所有文档（同一会话内联执行，避免嵌套会话）
        await session.execute(
            delete(FileDocModel).where(
                FileDocModel.kb_name.ilike(kb_file.kb_name),
                FileDocModel.file_name.ilike(kb_file.filename),
            )
        )

        # 异步查询关联的知识库并更新文件计数
        kb_result = await session.execute(
            select(KnowledgeBaseModel)
            .filter(KnowledgeBaseModel.kb_name.ilike(kb_file.kb_name))
        )
        kb = kb_result.scalars().first()
        if kb:
            kb.file_count -= 1

        await session.commit()

    return True


@with_async_session
async def delete_files_from_db(session, knowledge_base_name: str):
    await session.execute(
        delete(KnowledgeFileModel).where(KnowledgeFileModel.kb_name.ilike(knowledge_base_name))
    )
    await session.execute(
        delete(FileDocModel).where(FileDocModel.kb_name.ilike(knowledge_base_name))
    )

    result = await session.execute(
        select(KnowledgeBaseModel).where(KnowledgeBaseModel.kb_name.ilike(knowledge_base_name))
    )
    kb = result.scalars().first()
    if kb:
        kb.file_count = 0

    await session.commit()
    return True


@with_async_session
async def file_exists_in_db(session, kb_file: KnowledgeFile):
    result = await session.execute(
        select(KnowledgeFileModel)
        .filter(KnowledgeFileModel.file_name.ilike(kb_file.filename),
                KnowledgeFileModel.kb_name.ilike(kb_file.kb_name))
    )
    existing_file = result.scalars().first()
    return True if existing_file else False


@with_async_session
async def get_file_detail(session, kb_name: str, filename: str) -> dict:
    result = await session.execute(
        select(KnowledgeFileModel)
        .filter(KnowledgeFileModel.file_name.ilike(filename),
                KnowledgeFileModel.kb_name.ilike(kb_name))
    )
    file: KnowledgeFileModel = result.scalars().first()
    if file:
        return {
            "kb_name": file.kb_name,
            "file_name": file.file_name,
            "file_ext": file.file_ext,
            "file_version": file.file_version,
            "document_loader": file.document_loader_name,
            "text_splitter": file.text_splitter_name,
            "create_time": file.create_time,
            "file_mtime": file.file_mtime,
            "file_size": file.file_size,
            "custom_docs": file.custom_docs,
            "docs_count": file.docs_count,
        }
    else:
        return {}
