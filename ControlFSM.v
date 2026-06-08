`timescale 1ns / 1ps

module master_controller_fsm #(
    parameter LDM_AWIDTH = 9,
    parameter WGT_AWIDTH = 10
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    
    // SIMD Bus Outputs (Memory and Math Control)
    output reg                  ce_en,
    output reg                  ce_clear_acc,
    output reg                  ldm_we,
    output wire [LDM_AWIDTH-1:0] ldm_rd_addr,
    output wire [LDM_AWIDTH-1:0] ldm_wr_addr,
    output wire [2:0]            ldm_wr_lane_sel,
    output wire [WGT_AWIDTH-1:0] weight_addr,
    
    // Datapath & Halo Control (Added missing ports)
    output reg                  fsm_op_mode,     // 0 = Conv, 1 = Pool
    output reg                  fsm_relu_en,     // 1 = Apply ReLU
    output reg                  halo_latch,      // 1 = Catch neighbor data
    output reg                  halo_mux_sel,    // 1 = Read from Halo Reg
    
    // GAP & Dense Control (Added missing ports)
    output reg                  fsm_gap_we,
    output reg  [1:0]           fsm_gap_addr,
    output reg                  fsm_use_gap_buf,
    
    output reg                  done
);

// =========================================================================
    // 1. INSTRUCTION ROM DECODER (The model.py Memory Map)
    // =========================================================================
// =========================================================================
    // 1. INSTRUCTION ROM DECODER (The COMPLETE model.py Memory Map)
    // =========================================================================
    reg [5:0] layer_idx; // 6-bit counter to hold 44 Keras layers
    
    always @(*) begin
        // Default safe values to prevent latches
        op_type = OP_CONV; max_spatial = 4; max_filters = 8; max_steps = 7;
        base_rd_addr = 0; base_wr_addr = 0; alt_rd_addr = 0; base_wgt_addr = 0;
        inst_relu_en = 0; inst_halo_en = 0;

        case (layer_idx)
            // =================================================================
            // MACRO BLOCK 1: 160 Spatial Pixels (Assigned 4 per PE)
            // =================================================================
            // LAYER 0: mod1 = Conv1D(filters=8, kernel=7, strides=2)
            6'd0: begin op_type = OP_CONV; max_spatial = 4; max_filters = 8; max_steps = 7; base_rd_addr = 9'd0; base_wr_addr = 9'd100; base_wgt_addr = 10'd0; inst_relu_en = 1; inst_halo_en = 1; end
            
            // --- INCEPTION BLOCK 1A (Input: 100, Output: 140) ---
            6'd1: begin op_type = OP_POOL; max_spatial = 4; max_filters = 8; max_steps = 3; base_rd_addr = 9'd100; base_wr_addr = 9'd110; inst_halo_en = 1; end
            6'd2: begin op_type = OP_CONV; max_spatial = 4; max_filters = 2; max_steps = 1; base_rd_addr = 9'd110; base_wr_addr = 9'd140; end
            6'd3: begin op_type = OP_CONV; max_spatial = 4; max_filters = 2; max_steps = 1; base_rd_addr = 9'd100; base_wr_addr = 9'd120; end
            6'd4: begin op_type = OP_CONV; max_spatial = 4; max_filters = 2; max_steps = 3; base_rd_addr = 9'd120; base_wr_addr = 9'd141; inst_halo_en = 1; end
            6'd5: begin op_type = OP_CONV; max_spatial = 4; max_filters = 2; max_steps = 5; base_rd_addr = 9'd120; base_wr_addr = 9'd142; inst_halo_en = 1; end
            6'd6: begin op_type = OP_CONV; max_spatial = 4; max_filters = 2; max_steps = 7; base_rd_addr = 9'd120; base_wr_addr = 9'd143; inst_relu_en = 1; inst_halo_en = 1; end

            // --- INCEPTION BLOCK 1B (Input: 140, Output: 180) ---
            6'd7:  begin op_type = OP_POOL; max_spatial = 4; max_filters = 8; max_steps = 3; base_rd_addr = 9'd140; base_wr_addr = 9'd150; inst_halo_en = 1; end
            6'd8:  begin op_type = OP_CONV; max_spatial = 4; max_filters = 2; max_steps = 1; base_rd_addr = 9'd150; base_wr_addr = 9'd180; end
            6'd9:  begin op_type = OP_CONV; max_spatial = 4; max_filters = 2; max_steps = 1; base_rd_addr = 9'd140; base_wr_addr = 9'd160; end
            6'd10: begin op_type = OP_CONV; max_spatial = 4; max_filters = 2; max_steps = 3; base_rd_addr = 9'd160; base_wr_addr = 9'd181; inst_halo_en = 1; end
            6'd11: begin op_type = OP_CONV; max_spatial = 4; max_filters = 2; max_steps = 5; base_rd_addr = 9'd160; base_wr_addr = 9'd182; inst_halo_en = 1; end
            6'd12: begin op_type = OP_CONV; max_spatial = 4; max_filters = 2; max_steps = 7; base_rd_addr = 9'd160; base_wr_addr = 9'd183; inst_relu_en = 1; inst_halo_en = 1; end

            // LAYER 13: mod2 = Add()([mod1, incep1])
            6'd13: begin op_type = OP_ADD; max_spatial = 4; max_filters = 8; max_steps = 2; base_rd_addr = 9'd100; alt_rd_addr = 9'd180; base_wr_addr = 9'd200; end

            // =================================================================
            // MACRO BLOCK 2: 80 Spatial Pixels (Assigned 2 per PE)
            // =================================================================
            // LAYER 14: mod3 = Conv1D(filters=16, kernel=5, strides=2)
            6'd14: begin op_type = OP_CONV; max_spatial = 2; max_filters = 16; max_steps = 5; base_rd_addr = 9'd200; base_wr_addr = 9'd220; inst_relu_en = 1; inst_halo_en = 1; end

            // --- INCEPTION BLOCK 2A (Input: 220, Output: 260) ---
            6'd15: begin op_type = OP_POOL; max_spatial = 2; max_filters = 16; max_steps = 3; base_rd_addr = 9'd220; base_wr_addr = 9'd230; inst_halo_en = 1; end
            6'd16: begin op_type = OP_CONV; max_spatial = 2; max_filters = 4;  max_steps = 1; base_rd_addr = 9'd230; base_wr_addr = 9'd260; end
            6'd17: begin op_type = OP_CONV; max_spatial = 2; max_filters = 4;  max_steps = 1; base_rd_addr = 9'd220; base_wr_addr = 9'd240; end
            6'd18: begin op_type = OP_CONV; max_spatial = 2; max_filters = 4;  max_steps = 3; base_rd_addr = 9'd240; base_wr_addr = 9'd261; inst_halo_en = 1; end
            6'd19: begin op_type = OP_CONV; max_spatial = 2; max_filters = 4;  max_steps = 5; base_rd_addr = 9'd240; base_wr_addr = 9'd262; inst_halo_en = 1; end
            6'd20: begin op_type = OP_CONV; max_spatial = 2; max_filters = 4;  max_steps = 7; base_rd_addr = 9'd240; base_wr_addr = 9'd263; inst_relu_en = 1; inst_halo_en = 1; end

            // --- INCEPTION BLOCK 2B (Input: 260, Output: 300) ---
            6'd21: begin op_type = OP_POOL; max_spatial = 2; max_filters = 16; max_steps = 3; base_rd_addr = 9'd260; base_wr_addr = 9'd270; inst_halo_en = 1; end
            6'd22: begin op_type = OP_CONV; max_spatial = 2; max_filters = 4;  max_steps = 1; base_rd_addr = 9'd270; base_wr_addr = 9'd300; end
            6'd23: begin op_type = OP_CONV; max_spatial = 2; max_filters = 4;  max_steps = 1; base_rd_addr = 9'd260; base_wr_addr = 9'd280; end
            6'd24: begin op_type = OP_CONV; max_spatial = 2; max_filters = 4;  max_steps = 3; base_rd_addr = 9'd280; base_wr_addr = 9'd301; inst_halo_en = 1; end
            6'd25: begin op_type = OP_CONV; max_spatial = 2; max_filters = 4;  max_steps = 5; base_rd_addr = 9'd280; base_wr_addr = 9'd302; inst_halo_en = 1; end
            6'd26: begin op_type = OP_CONV; max_spatial = 2; max_filters = 4;  max_steps = 7; base_rd_addr = 9'd280; base_wr_addr = 9'd303; inst_relu_en = 1; inst_halo_en = 1; end

            // LAYER 27: mod4 = Add()([mod3, incep2])
            6'd27: begin op_type = OP_ADD; max_spatial = 2; max_filters = 16; max_steps = 2; base_rd_addr = 9'd220; alt_rd_addr = 9'd300; base_wr_addr = 9'd320; end

            // =================================================================
            // MACRO BLOCK 3: 40 Spatial Pixels (Assigned 1 per PE)
            // =================================================================
            // LAYER 28: mod5 = Conv1D(filters=32, kernel=3, strides=2)
            6'd28: begin op_type = OP_CONV; max_spatial = 1; max_filters = 32; max_steps = 3; base_rd_addr = 9'd320; base_wr_addr = 9'd340; inst_relu_en = 1; inst_halo_en = 1; end

            // --- INCEPTION BLOCK 3A (Input: 340, Output: 380) ---
            6'd29: begin op_type = OP_POOL; max_spatial = 1; max_filters = 32; max_steps = 3; base_rd_addr = 9'd340; base_wr_addr = 9'd350; inst_halo_en = 1; end
            6'd30: begin op_type = OP_CONV; max_spatial = 1; max_filters = 8;  max_steps = 1; base_rd_addr = 9'd350; base_wr_addr = 9'd380; end
            6'd31: begin op_type = OP_CONV; max_spatial = 1; max_filters = 8;  max_steps = 1; base_rd_addr = 9'd340; base_wr_addr = 9'd360; end
            6'd32: begin op_type = OP_CONV; max_spatial = 1; max_filters = 8;  max_steps = 3; base_rd_addr = 9'd360; base_wr_addr = 9'd381; inst_halo_en = 1; end
            6'd33: begin op_type = OP_CONV; max_spatial = 1; max_filters = 8;  max_steps = 5; base_rd_addr = 9'd360; base_wr_addr = 9'd382; inst_halo_en = 1; end
            6'd34: begin op_type = OP_CONV; max_spatial = 1; max_filters = 8;  max_steps = 7; base_rd_addr = 9'd360; base_wr_addr = 9'd383; inst_relu_en = 1; inst_halo_en = 1; end

            // --- INCEPTION BLOCK 3B (Input: 380, Output: 420) ---
            6'd35: begin op_type = OP_POOL; max_spatial = 1; max_filters = 32; max_steps = 3; base_rd_addr = 9'd380; base_wr_addr = 9'd390; inst_halo_en = 1; end
            6'd36: begin op_type = OP_CONV; max_spatial = 1; max_filters = 8;  max_steps = 1; base_rd_addr = 9'd390; base_wr_addr = 9'd420; end
            6'd37: begin op_type = OP_CONV; max_spatial = 1; max_filters = 8;  max_steps = 1; base_rd_addr = 9'd380; base_wr_addr = 9'd400; end
            6'd38: begin op_type = OP_CONV; max_spatial = 1; max_filters = 8;  max_steps = 3; base_rd_addr = 9'd400; base_wr_addr = 9'd421; inst_halo_en = 1; end
            6'd39: begin op_type = OP_CONV; max_spatial = 1; max_filters = 8;  max_steps = 5; base_rd_addr = 9'd400; base_wr_addr = 9'd422; inst_halo_en = 1; end
            6'd40: begin op_type = OP_CONV; max_spatial = 1; max_filters = 8;  max_steps = 7; base_rd_addr = 9'd400; base_wr_addr = 9'd423; inst_relu_en = 1; inst_halo_en = 1; end

            // LAYER 41: Final Add()([mod5, incep3])
            6'd41: begin op_type = OP_ADD; max_spatial = 1; max_filters = 32; max_steps = 2; base_rd_addr = 9'd340; alt_rd_addr = 9'd420; base_wr_addr = 9'd450; end

            // =================================================================
            // MACRO BLOCK 4: GLOBAL REDUCTION & DENSE
            // =================================================================
            // LAYER 42: Global Average Pooling (GAP)
            // Pulls the final 32 channels and fires them into the GAP Tree
            6'd42: begin op_type = OP_GAP; max_spatial = 1; max_filters = 32; max_steps = 4; base_rd_addr = 9'd450; base_wr_addr = 9'd0; end

            // LAYER 43: Dense(5) Output Logits
            // PE 0 reads from gap_ram and computes the final 5 classification scores
            6'd43: begin op_type = OP_DENSE; max_spatial = 1; max_filters = 5; max_steps = 4; base_rd_addr = 9'd0; base_wr_addr = 9'd500; end
            
            default: op_type = OP_CONV;
        endcase
    end

    // =========================================================================
    // 2. THE DATAPATH ADDRESS ROUTING
    // =========================================================================
    reg [2:0] spatial_cnt;
    reg [4:0] filter_cnt;
    reg [2:0] step_cnt;

    assign ldm_wr_lane_sel = filter_cnt[2:0];
    assign ldm_wr_addr = base_wr_addr + (spatial_cnt * (max_filters >> 3)) + (filter_cnt >> 3);
    assign weight_addr = base_wgt_addr + (filter_cnt * max_steps) + step_cnt;
    
    // For Add layer, toggle between 'mod' address and 'incep' address. Otherwise standard slide.
    reg [LDM_AWIDTH-1:0] alt_rd_addr; // NEW: Holds the skip connection pointer

    // For Add layer, toggle between 'mod' (base) and 'incep' (alt) address.
    assign ldm_rd_addr = (op_type == OP_ADD && step_cnt == 1) ? (alt_rd_addr + spatial_cnt) : 
                         base_rd_addr + (spatial_cnt * max_steps) + step_cnt;

    // =========================================================================
    // 3. THE MICROCODE STATE MACHINE
    // =========================================================================
    reg [2:0] state;
    localparam IDLE = 0, FETCH = 1, HALO_PREP = 2, EXECUTE = 3, GAP_EXECUTE = 4, FINISH = 5;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            layer_idx <= 0; spatial_cnt <= 0; filter_cnt <= 0; step_cnt <= 0;
            ce_en <= 0; ce_clear_acc <= 0; ldm_we <= 0; done <= 0;
            fsm_gap_we <= 0; fsm_gap_addr <= 0; fsm_use_gap_buf <= 0;
            halo_latch <= 0; halo_mux_sel <= 0; fsm_op_mode <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0; layer_idx <= 0;
                    if (start) state <= FETCH;
                end

                FETCH: begin
                    // 1. Decode instruction and set static datapath flags
                    fsm_relu_en <= inst_relu_en;
                    fsm_op_mode <= (op_type == OP_POOL) ? 1'b1 : 1'b0;
                    fsm_use_gap_buf <= (op_type == OP_DENSE) ? 1'b1 : 1'b0;
                    spatial_cnt <= 0; filter_cnt <= 0; step_cnt <= 0;
                    
                    if (layer_idx == 44) state <= FINISH; // ALL 44 LAYERS COMPLETE!
                    else if (op_type == OP_GAP) state <= GAP_EXECUTE;
                    else if (inst_halo_en) state <= HALO_PREP;
                    else state <= EXECUTE;
                end

                HALO_PREP: begin
                    // Trigger the 1-cycle boundary pixel exchange
                    halo_latch <= 1'b1; 
                    state <= EXECUTE;
                end

                EXECUTE: begin
                    halo_latch <= 1'b0;
                    ce_en <= 1'b1;
                    
                    // Dynamic Halo Routing: If we are looking at pixel t-1 on the left PE boundary
                    halo_mux_sel <= (inst_halo_en && spatial_cnt == 0 && step_cnt == 0) ? 1'b1 : 1'b0;

                    if (step_cnt == 0) ce_clear_acc <= 1'b1;
                    else               ce_clear_acc <= 1'b0;

                    if (step_cnt == max_steps - 1) begin
                        ldm_we <= 1'b1; 
                        step_cnt <= 0;

                        if (filter_cnt == max_filters - 1) begin
                            filter_cnt <= 0;
                            if (spatial_cnt == max_spatial - 1) begin
                                // Layer finished! Go fetch the next layer.
                                ce_en <= 0; ldm_we <= 0;
                                layer_idx <= layer_idx + 1;
                                state <= FETCH;
                            end else begin
                                spatial_cnt <= spatial_cnt + 1;
                            end
                        end else begin
                            filter_cnt <= filter_cnt + 1;
                        end
                    end else begin
                        ldm_we <= 1'b0;
                        step_cnt <= step_cnt + 1;
                    end
                end

                GAP_EXECUTE: begin
                    // Stream the 32 channels through the GAP Tree and into gap_ram
                    fsm_gap_we <= 1'b1;
                    fsm_gap_addr <= step_cnt[1:0];
                    
                    if (step_cnt == 3) begin // 32 channels / 8 lanes = 4 cycles
                        fsm_gap_we <= 0;
                        layer_idx <= layer_idx + 1;
                        state <= FETCH;
                    end else begin
                        step_cnt <= step_cnt + 1;
                    end
                end

                FINISH: begin
                    ce_en <= 0; ldm_we <= 0; done <= 1'b1; 
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule