package com.dicoding.capstone_newschecker.remote

import com.dicoding.capstone_newschecker.data.NewsPrediction
import retrofit2.http.POST
import retrofit2.http.Query

interface ApiService {
    @POST("predict")
    suspend fun predictNews(@Query("title") title: String): NewsPrediction
}
