package com.dicoding.capstone_newschecker.response

import com.dicoding.capstone_newschecker.data.NewsData

data class Response(
    val data: NewsData?,
    val message: String?
)
